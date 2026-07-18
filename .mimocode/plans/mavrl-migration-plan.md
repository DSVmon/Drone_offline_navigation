# План миграции на архитектуру MAVRL

## Краткое описание

Полная адаптация архитектуры MAVRL (TU Delft, IEEE RA-L 2025) для проекта Drone Offline Navigation. MAVRL решает ту же задачу — автономный обход препятствий дроном через stereo vision — но с радикально более эффективным подходом.

## Текущая архитектура → Целевая архитектура

| Компонент | Текущий | MAVRL (целевой) |
|-----------|---------|-----------------|
| Вход (image) | 5 чисел (min distance) | Depth map 256×256 |
| Вход (state) | 14-dim (stereo + pos + vel + orient) | 7-dim (goal-oriented) |
| Encoder | Нет | CNN → 64-dim latent |
| Memory | Нет | LSTM (256 hidden) |
| Policy | MLP [256,256] | CNN + LSTM + MLP [256,256] |
| Value net | MLP [256,256] | CNN + LSTM + MLP [512,512] |
| Action | 3-dim (vx, vz, yaw) | 4-dim (ax, ay, az, yaw_rate) |
| Goal | Нет (лети вперёд) | Goal-point (start → goal) |
| RL algorithm | PPO (SB3) | RecurrentPPO (custom) |
| Training | BC→PPO | PPO→collect→VAE→LSTM→retrain PPO |
| Speed | Фиксированный | Adaptive (varying speed) |

---

## Этап 1: Depth Map пайплайн (КРИТИЧЕСКИЙ)

**Цель:** Заменить 5 чисел stereo distances на полноценную depth map 256×256.

### 1.1. Изменения в navigation_node.cpp

Файл: `src/drone_navigation/src/navigation_node.cpp`

**Текущее:** StereoBM выдаёт 5 min-distance значений.
**Целевое:** StereoBM выдаёт depth map в виде `sensor_msgs/Image`.

Изменения:
- Добавить publisher `/navigation_node/depth_map` типа `sensor_msgs/Image` (format: `32FC1`)
- Depth map = `stereo_model_.getZ(disp_visible)` для каждого пикселя
- Ограничить диапазон 0.1–12.0м (как в MAVRL)
- Resize до 256×256 через `cv::pyrDown` + `cv::resize`
- Оставить старый publisher `stereo_distances` для обратной совместимости

```cpp
// Новый publisher
depth_map_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/depth_map", 10);

// В vioCallback, после disparity:
cv::Mat depth_map_32f;
disp_visible.copyTo(depth_map_32f);
// Конвертируем disparity → depth (Z = f*B/d)
for (int v = 0; v < depth_map_32f.rows; v++) {
    for (int u = 0; u < depth_map_32f.cols; u++) {
        float disp = depth_map_32f.at<float>(v, u);
        if (disp > 0.1f) {
            depth_map_32f.at<float>(v, u) = stereo_model_.getZ(disp);
        } else {
            depth_map_32f.at<float>(v, u) = 12.0f; // far
        }
    }
}
// Clamp и resize
cv::threshold(depth_map_32f, depth_map_32f, 12.0, 12.0, cv::THRESH_TRUNC);
cv::resize(depth_map_32f, depth_map_32f, cv::Size(256, 256));
// Publish
auto depth_msg = cv_bridge::CvImage(header, "32FC1", depth_map_32f).toImageMsg();
depth_map_pub_->publish(*depth_msg);
```

### 1.2. Изменения в drone_env.py

Файл: `learning/drone_env.py`

- Добавить subscriber на `/navigation_node/depth_map` типа `sensor_msgs/Image`
- Preprocess: clamp 0-12м, нормализация в 0-255, resize до 256×256
- Observation space: `Dict{'image': Box(0, 255, (1, 256, 256)), 'state': Box(-inf, inf, (7,))}`
- Добавить image memory buffer (последние N depth maps)

```python
# Новый observation space
self.observation_space = spaces.Dict({
    'image': spaces.Box(low=0, high=255, shape=(1, 256, 256), dtype=np.uint8),
    'state': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
})

# Subscriber
self._sub_depth = self.node.create_subscription(
    Image, '/navigation_node/depth_map', self._depth_callback, 10
)

def _depth_callback(self, msg):
    depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough') / 1000.0  # mm → m
    depth = np.clip(depth, 0.1, 12.0)
    depth = (depth / 12.0 * 255.0).astype(np.uint8)
    depth = cv2.resize(depth, (256, 256))
    with self._lock:
        self.depth_image = depth
```

### 1.3. Изменения в inference_node.py

Файл: `learning/inference_node.py`

Те же изменения — subscriber на depth map, preprocess, Dict observation.

---

## Этап 2: Goal-point навигация (КРИТИЧЕСКИЙ)

**Цель:** Дрон получает конкретную точку назначения (конец тоннеля) и летит к ней.

### 2.1. Определение goal-point

Файл: `learning/drone_env.py`

Goal-point = конец тоннеля (по оси X). Определяется при генерации пещеры:
- Длина пещеры определяется из cave generator
- Goal = (cave_length × 0.8, 0, 1.75) — 80% длины, центр по Y, высота 1.75м

```python
def reset(self, seed=None, options=None):
    # ... existing reset code ...
    
    # Определяем goal-point из длины пещеры
    # straight_cave: 100m → goal_x = 80
    # gentle_cave: 100m → goal_x = 80
    # procedural_cave: 100m → goal_x = 80
    cave_length = 100.0  # из параметров генератора
    self.goal_point = np.array([cave_length * 0.8, 0.0, 1.75])
    
    # Entrance heading (направление полёта)
    self.entrance_heading = (
        math.cos(self.current_yaw),
        math.sin(self.current_yaw),
    )
```

### 2.2. State space (7-dim, как в MAVRL)

Файл: `learning/drone_env.py`

Заменить текущий 14-dim state на 7-dim goal-oriented state:

```python
def _build_observation(self):
    with self._lock:
        pos = np.array([self.current_x, self.current_y, self.current_z])
        vel = np.array([self.odom_vx, self.odom_vy, self.odom_vz])
    
    # MAVRL-style state (7-dim)
    delta_p = self.goal_point - pos
    log_distance = np.log(np.sqrt(delta_p[0]**2 + delta_p[1]**2) + 1.0)
    
    # Body-frame velocity
    vel_body = self._world2body(vel)
    horizon_vel = np.sqrt(vel_body[0]**2 + vel_body[1]**2)
    
    # Angle to goal
    theta = np.arctan2(-delta_p[0], delta_p[1])
    horizon_vel_dire = np.arctan2(vel_body[1], vel_body[0])
    
    state = np.array([
        log_distance,           # log(dist_to_goal)
        horizon_vel,            # горизонтальная скорость
        theta,                  # угол до цели
        horizon_vel_dire,       # направление горизонтальной скорости
        delta_p[2],             # разница высот до цели
        vel_body[2],            # вертикальная скорость (body)
        self.current_yaw,       # yaw
    ], dtype=np.float64)
    
    return {'image': self.depth_image, 'state': state}

def _world2body(self, world_vel):
    """Конвертация world-frame velocity в body-frame."""
    cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)
    # FLU to RFU conversion (как в MAVRL)
    body_x = -world_vel[1]  # RFU
    body_y = world_vel[0]
    body_z = world_vel[2]
    # Rotate by -yaw
    rot_body_x = cy * body_x + sy * body_y
    rot_body_y = -sy * body_x + cy * body_y
    return np.array([rot_body_x, rot_body_y, body_z])
```

### 2.3. Completion detection

Файл: `learning/drone_env.py`

```python
def _check_completion(self):
    """Проверка достижения цели."""
    dist_to_goal = np.linalg.norm(
        np.array([self.current_x, self.current_y, self.current_z]) - self.goal_point
    )
    return dist_to_goal < 2.0  # Within 2m of goal
```

---

## Этап 3: Action Space (4-dim, как в MAVRL)

**Цель:** Перейти от velocity commands к body-frame accelerations.

### 3.1. Action space

Файл: `learning/config.py`

```python
# Было: 3-dim (vx, vz, yaw)
# Стало: 4-dim (ax, ay, az, yaw_rate) — body-frame accelerations
ACTION_DIM = 4
ACTION_ACC_MAX = np.array([3.0, 3.0, 3.0])  # m/s² body-frame
ACTION_YAW_RATE_MAX = 2.0  # rad/s

# Action normalization
ACTION_MEAN = np.array([0.0, 0.0, 0.0, 0.0])
ACTION_STD = np.array([3.0, 3.0, 3.0, 2.0])
```

### 3.2. Action application

Файл: `learning/drone_env.py`

```python
def _apply_action(self, action):
    """Denormalize action и применить как body-frame acceleration."""
    # Denormalize: action [-1,1] → physical units
    cmd = action * self.action_std + self.action_mean
    
    acc_body = cmd[:3]  # body-frame acceleration
    yaw_rate = cmd[3]
    
    # Integrate: vel_world = vel_world + R(body→world) * acc_body * dt
    acc_world = self._body2world(acc_body)
    self.odom_vx += acc_world[0] * config.DT
    self.odom_vy += acc_world[1] * config.DT
    self.odom_vz += acc_world[2] * config.DT
    
    # Publish as velocity command
    msg = Twist()
    msg.linear.x = self.odom_vx
    msg.linear.y = self.odom_vy
    msg.linear.z = self.odom_vz
    msg.angular.z = yaw_rate
    self._cmd_vel_pub.publish(msg)
    
    # Z-axis via Gazebo service
    self._set_gazebo_z(self.odom_vz)
```

### 3.3. Body-to-world transformation

```python
def _body2world(self, acc_body):
    """Body-frame acceleration → world-frame."""
    cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)
    # RFU → FLU
    flu = np.array([acc_body[1], -acc_body[0], acc_body[2]])
    # Rotate
    world_flu = np.array([
        cy * flu[0] - sy * flu[1],
        sy * flu[0] + cy * flu[1],
        flu[2]
    ])
    # FLU → RFU
    return np.array([-world_flu[1], world_flu[0], world_flu[2]])
```

---

## Этап 4: CNN + LSTM Policy (КРИТИЧЕСКИЙ)

**Цель:** Заменить MLP на CNN+LSTM архитектуру.

### 4.1. Политика (новый файл)

Файл: `learning/policy.py` (НОВЫЙ)

```python
import torch
import torch.nn as nn
from torch.distributions import Normal

class DepthEncoder(nn.Module):
    """CNN для кодирования depth map 256×256 → 64-dim latent."""
    def __init__(self, features_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4, padding=2),  # 256→64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64→32
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 32→32
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64 * 32 * 32 = 65536
        self.fc = nn.Linear(65536, features_dim)
        self.fc_logsigma = nn.Linear(65536, features_dim)  # VAE
        
    def forward(self, x):
        h = self.cnn(x)
        mu = self.fc(h)
        logsigma = self.fc_logsigma(h)
        return mu, logsigma


class RecurrentPolicy(nn.Module):
    """CNN + LSTM + Actor-Critic policy."""
    def __init__(self, features_dim=64, lstm_hidden=256, act_dim=4):
        super().__init__()
        # Image encoder
        self.encoder = DepthEncoder(features_dim)
        
        # State encoder
        self.state_fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=features_dim + 64,  # latent + state
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_mean = nn.Linear(256, act_dim)
        self.action_log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
        self.lstm_hidden = lstm_hidden
    
    def forward(self, image, state, lstm_hidden=None):
        """
        Args:
            image: (batch, seq, 1, 256, 256)
            state: (batch, seq, 7)
            lstm_hidden: ((1, batch, 256), (1, batch, 256))
        """
        B, T = image.shape[:2]
        
        # Encode images
        img_flat = image.reshape(B * T, 1, 256, 256)
        mu, logsigma = self.encoder(img_flat)
        latent = mu  # или reparameterize
        
        # Encode state
        state_flat = state.reshape(B * T, 7)
        state_enc = self.state_fc(state_flat)
        
        # Combine
        features = torch.cat([latent, state_enc], dim=-1)
        features = features.reshape(B, T, -1)
        
        # LSTM
        if lstm_hidden is None:
            lstm_out, hidden = self.lstm(features)
        else:
            lstm_out, hidden = self.lstm(features, lstm_hidden)
        
        # Actor
        actor_features = self.actor(lstm_out)
        action_mean = self.action_mean(actor_features)
        action_std = self.action_log_std.exp().expand_as(action_mean)
        
        # Critic
        value = self.critic(lstm_out)
        
        return action_mean, action_std, value, hidden
    
    def predict(self, image, state, lstm_hidden=None, deterministic=False):
        with torch.no_grad():
            mean, std, value, hidden = self.forward(image, state, lstm_hidden)
            mean = mean[:, -1]  # last timestep
            std = std[:, -1]
            value = value[:, -1]
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
            
            action = torch.tanh(action)  # clip to [-1, 1]
            return action.cpu().numpy(), hidden
```

### 4.2. RecurrentPPO (новый файл)

Файл: `learning/recurrent_ppo.py` (НОВЫЙ)

Адаптация RecurrentPPO из MAVRL. Ключевые отличия от стандартного PPO:
- LSTM hidden states в rollout buffer
- Sequence-based batching
- Bootstrap value для truncated episodes

---

## Этап 5: VAE (Variational AutoEncoder)

**Цель:** Предобучить encoder на depth maps для лучшего представления.

### 5.1. VAE model

Файл: `learning/vae.py` (НОВЫЙ)

```python
class DepthVAE(nn.Module):
    """VAE для depth maps 256×256 → 64-dim latent."""
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder (тот же что в DepthEncoder)
        self.encoder = DepthEncoder(latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 65536),
            nn.ReLU(),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 8, stride=4, padding=2),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = self._reparameterize(mu, logsigma)
        recon = self.decoder(z)
        return recon, mu, logsigma
    
    def _reparameterize(self, mu, logsigma):
        if self.training:
            std = logsigma.exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
```

### 5.2. VAE training

Файл: `learning/train_vae.py` (НОВЫЙ)

Pipeline:
1. Запустить simulation с control_node (реактивный контроллер)
2. Собрать 10K-50K depth images
3. Обучить VAE (reconstruction loss + KL divergence)

---

## Этап 6: Training Pipeline

**Цель:** 4-этапный pipeline как в MAVRL.

### 6.1. Этапы обучения

```
Этап A: Train initial PPO (без препятствий)
  → Straight cave, simple environment
  → CNN random init, LSTM random init
  → 200K steps
  
Этап B: Collect depth data
  → Использовать control_node (реактивный контроллер)
  → Собрать 50K depth image sequences
  → Сохранить в .npz файлы
  
Этап C: Train VAE + LSTM
  → VAE: на collected depth data (reconstruction)
  → LSTM: предсказание следующего depth по истории
  → Freeze encoder weights
  
Этап D: Retrain PPO with frozen encoder
  → CNN encoder frozen (from VAE)
  → LSTM frozen (from step C)
  → Train only actor + critic heads
  → 2M steps, curriculum learning
```

### 6.2. Изменения в train.py

Файл: `learning/train.py`

```python
def main():
    parser.add_argument("--stage", type=str, default="all",
                       choices=["a", "b", "c", "d", "all"])
    
    if args.stage in ["a", "all"]:
        stage_a_initial_ppo()
    
    if args.stage in ["b", "all"]:
        stage_b_collect_data()
    
    if args.stage in ["c", "all"]:
        stage_c_train_vae_lstm()
    
    if args.stage in ["d", "all"]:
        stage_d_retrain_ppo()
```

### 6.3. Callbacks

Файл: `learning/callbacks.py`

Обновить:
- Success detection: dist_to_goal < 2.0m
- Curriculum: radius obstacles (как в MAVRL)
- TensorBoard: log depth maps, latent vectors, LSTM states

---

## Этап 7: Curriculum Learning

**Цель:** Параметрический curriculum как в MAVRL.

### 7.1. Параметры среды

Файл: `learning/config.py`

```python
# Curriculum через параметры пещеры
CURRICULUM_STAGES = [
    {
        "name": "easy",
        "cave_width": 8.0,        # широкий тоннель
        "turn_angle_max": 0,       # без поворотов
        "obstacle_density": 0.0,   # без препятствий
        "radius_factor": 1.0,      # размер зоны обзора
    },
    {
        "name": "medium",
        "cave_width": 5.0,
        "turn_angle_max": 30,      # плавные повороты ±30°
        "obstacle_density": 0.3,   # лёгкие препятствия
        "radius_factor": 0.7,
    },
    {
        "name": "hard",
        "cave_width": 3.5,
        "turn_angle_max": 55,      # крутые повороты ±55°
        "obstacle_density": 0.6,   # плотные препятствия
        "radius_factor": 0.5,
    },
]
```

### 7.2. Динамическое изменение среды

Файл: `learning/utils.py`

```python
def change_cave_params(stage):
    """Изменить параметры пещеры для curriculum."""
    # Генерировать новую пещеру с параметрами stage
    # Не перезапускать Gazebo, а менять через set_entity_state
    pass
```

---

## Этап 8: Adaptive Speed

**Цель:** Адаптивная скорость к сложности среды.

### 8.1. Speed-dependent reward

Файл: `learning/reward.py`

```python
# Бонус за скорость зависит от расстояния до препятствий
min_dist = min(distances)
if min_dist > 3.0:
    speed_bonus = 0.03 * vx  # быстро в открытом пространстве
elif min_dist > 1.5:
    speed_bonus = 0.01 * vx  # медленнее
else:
    speed_bonus = -0.02 * vx  # штраф за скорость в тесном месте
```

---

## Этап 9: Вспомогательные файлы

### 9.1. collect_data.py

Файл: `learning/collect_data.py` (НОВЫЙ)

Сбор depth image sequences для обучения VAE+LSTM:
```python
# Подписка на /navigation_node/depth_map + /odom + /goal_point
# Сохранение в формате: {'images': (N, 256, 256), 'states': (N, 7)}
```

### 9.2. test_ppo.py

Файл: `learning/test_ppo.py` (НОВЫЙ)

Визуализация полёта с обученной моделью:
```python
# Загрузка модели
# Запуск inference
# Запись видео с depth map + trajectory
```

---

## Порядок реализации

| # | Этап | Файлы | Сложность | Время |
|---|------|-------|-----------|-------|
| 1 | Depth Map pipeline | navigation_node.cpp, drone_env.py | Средняя | 2-3 дня |
| 2 | Goal-point навигация | drone_env.py, config.py | Низкая | 1 день |
| 3 | Action space (4-dim) | drone_env.py, config.py | Средняя | 1-2 дня |
| 4 | CNN + LSTM policy | policy.py, recurrent_ppo.py | Высокая | 3-5 дней |
| 5 | VAE | vae.py, train_vae.py | Средняя | 2-3 дня |
| 6 | Training pipeline | train.py, callbacks.py | Средняя | 2-3 дня |
| 7 | Curriculum | config.py, utils.py | Низкая | 1 день |
| 8 | Adaptive speed | reward.py | Низкая | 0.5 дня |
| 9 | Testing & debug | test_ppo.py, inference_node.py | Средняя | 2-3 дня |

**Общее время:** ~15-20 дней

---

## Критические файлы для изменения

| Файл | Приоритет | Изменения |
|------|-----------|-----------|
| `src/drone_navigation/src/navigation_node.cpp` | Критический | Добавить depth map publisher |
| `learning/drone_env.py` | Критический | Dict obs, goal-point, 7-dim state, depth subscriber |
| `learning/config.py` | Критический | Action dim, curriculum params, goal params |
| `learning/policy.py` | Критический | НОВЫЙ: CNN + LSTM + Actor-Critic |
| `learning/recurrent_ppo.py` | Критический | НОВЫЙ: RecurrentPPO |
| `learning/vae.py` | Высокий | НОВЫЙ: Depth VAE |
| `learning/train_vae.py` | Высокий | НОВЫЙ: VAE training |
| `learning/train.py` | Высокий | 4-stage pipeline |
| `learning/collect_data.py` | Высокий | НОВЫЙ: depth data collection |
| `learning/reward.py` | Средний | Goal reward, adaptive speed |
| `learning/inference_node.py` | Средний | Dict obs, 4-dim action |
| `learning/test_ppo.py` | Низкий | НОВЫЙ: visualization |
| `learning/callbacks.py` | Низкий | Success detection, curriculum |
| `learning/utils.py` | Низкий | Dynamic cave params |
