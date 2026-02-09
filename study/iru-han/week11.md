# 휴머노이드 로봇
## 관절 구조 설계 (Joint Structure Design)

### 1. 하체 (Legs): 12개 (보행의 핵심)
다리 하나당 6개의 관절이 있어 사람처럼 무릎을 굽히고 발목을 돌릴 수 있음.

- **고관절 (Hip)**: 3개 × 2 (Pitch, Roll, Yaw) → 다리를 앞뒤/좌우/회전  
- **무릎 (Knee)**: 1개 × 2 (Pitch) → 굽히기  
- **발목 (Ankle)**: 2개 × 2 (Pitch, Roll) → 지면 접지력 유지  

---

### 2. 허리 (Waist): 3개 (균형의 핵심)
상체의 무게중심을 이동시켜 넘어짐을 방지하는 척추 역할.

- **허리 (Waist)**: 3개 (Yaw, Roll, Pitch) → 몸통 회전 및 기울기 제어  

---

### 3. 상체 (Arms): 14개 (무게중심 보정)
걸을 때 팔을 흔들어 **각운동량(Angular Momentum)**을 상쇄함.

- **어깨 (Shoulder)**: 3개 × 2 (Pitch, Roll, Yaw)  
- **팔꿈치 (Elbow)**: 1개 × 2 (Pitch)  
- **손목 (Wrist)**: 3개 × 2 (Roll, Pitch, Yaw)  


# 4족 보행로봇과의 비교
A. 역기구학 (Inverse Kinematics) 구조 변경 필요 <br>
참고한 SpotMicro 코드는 다리 관절이 3개(hip, thigh, calf)뿐이지만, G1은 다리 하나에 6개의 관절이 있어 훨씬 복잡한 3차원 회전 행렬 계산이 필요함.<br>

B. 보행 패턴 생성 (Gait Generator)<br>
SpotMicro는 4개의 다리가 박자에 맞춰 움직이는 **트로팅(Trotting)**을 구현했으나, 이를 2족 보행에 맞게 **교차 보행(Cross Walking)**으로 재설계 해야함.<br>

C. G1은 다리가 2개라 훨씬 정교한 무게중심 제어가 필요하다는 것을 깨닫게 됨.<br>
<br><br>
현재 구현된 코드는 정적 보행(Static Walking) 기반으로 작성되어 있어, 로봇이 보행을 지속할수록 상체가 앞으로 기울어지며 넘어지는 현상이 발생<br>

<img width="111" height="145" alt="image" src="https://github.com/user-attachments/assets/31f96e26-0f47-4085-9a43-33120720e66c" />

```
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from math import sqrt, atan2, acos, pi, sin, cos


# =========================================================
# [발표 설명 포인트 1] 역기구학 (Inverse Kinematics)
# 발 끝의 좌표(x,y,z)를 주면 -> 모터가 움직여야 할 각도를 계산합니다.
# =========================================================
class G1Kinematics:
    def __init__(self):
        # 로봇의 기구학적 정보 (다리 링크 길이)
        self.l1 = 0.30  # 허벅지 길이 (m)
        self.l2 = 0.30  # 종아리 길이 (m)

    def leg_IK(self, x, y, z):
        # 1. 목표 지점까지의 거리 계산
        dist = sqrt(x ** 2 + y ** 2 + z ** 2)
        # 물리적으로 닿을 수 없는 거리는 최대 길이로 제한 (에러 방지)
        dist = np.clip(dist, 0.1, self.l1 + self.l2 - 0.01)

        # 2. 제2 코사인 법칙을 이용해 무릎 관절 각도 계산
        cos_knee = (self.l1 ** 2 + self.l2 ** 2 - dist ** 2) / (2 * self.l1 * self.l2)
        theta_knee = pi - acos(np.clip(cos_knee, -1.0, 1.0))

        # 3. 고관절(Hip) Pitch 각도 계산
        alpha = acos(np.clip((self.l1 ** 2 + dist ** 2 - self.l2 ** 2) / (2 * self.l1 * dist), -1.0, 1.0))
        theta_hip_pitch = atan2(x, -z) - alpha

        # 4. 고관절 Roll 및 발목 보정 (발바닥 수평 유지)
        theta_hip_roll = atan2(y, -z)
        theta_ankle_pitch = -theta_hip_pitch - theta_knee
        theta_ankle_roll = -theta_hip_roll

        # 계산된 6개의 관절 각도 반환
        return [theta_hip_pitch, theta_hip_roll, 0.0, theta_knee, theta_ankle_pitch, theta_ankle_roll]


# =========================================================
# [발표 설명 포인트 2] 보행 궤적 생성기 (Gait Generator)
# 시간(t)에 따라 로봇 발이 그려야 할 사인파(Sine-wave) 궤적을 만듭니다.
# =========================================================
class G1Gait:
    def __init__(self):
        self.step_height = 0.10  # 발을 드는 높이 (cm) -> 잘 보이게 높임
        self.step_length = 0.20  # 보폭 (cm) -> 시원시원하게 걷도록 설정
        self.cycle_time = 1.0  # 한 걸음 걸리는 시간 (1초)
        self.default_z = -0.65  # 다리를 적당히 뻗은 기본 높이

    def get_joint_targets(self, t):
        # 0.0 ~ 1.0 사이의 보행 위상(Phase) 계산
        phase = (t % self.cycle_time) / self.cycle_time

        # 발의 기본 위치 설정 (골반 기준)
        l_pos = [0.0, 0.08, self.default_z]  # 왼발
        r_pos = [0.0, -0.08, self.default_z]  # 오른발

        # 사인(Sin) 함수를 이용한 교차 보행 궤적 생성
        if phase < 0.5:
            # [왼발 스윙 구간]
            p = phase / 0.5
            l_pos[2] += self.step_height * sin(pi * p)  # 위로 들기
            l_pos[0] += (p - 0.5) * self.step_length  # 앞으로 내밀기
            # 오른발은 뒤로 밀어주기 (상대 속도)
            r_pos[0] -= (p - 0.5) * self.step_length
        else:
            # [오른발 스윙 구간]
            p = (phase - 0.5) / 0.5
            r_pos[2] += self.step_height * sin(pi * p)
            r_pos[0] += (p - 0.5) * self.step_length
            l_pos[0] -= (p - 0.5) * self.step_length

        return l_pos, r_pos


# =========================================================
# [발표 설명 포인트 3] 시뮬레이션 및 PID 제어
# 로봇이 넘어지지 않도록 '가상의 힘'을 적용해 공중에 고정합니다.
# =========================================================
class G1LevitationSim:
    def __init__(self, xml_path):
        # XML 파일 경로 안전장치
        if not os.path.exists(xml_path):
            # 혹시 경로가 다를 경우를 대비한 예비 경로
            xml_path = "scene.xml"
            if not os.path.exists(xml_path):
                # 최후의 수단: 상대 경로
                xml_path = "../description/scene.xml"

        # MuJoCo 모델 로드
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.kin = G1Kinematics()
        self.gait = G1Gait()

        # [중요] 로봇의 총 질량 자동 계산 (약 12~13kg) -> 중력 보상에 사용
        self.total_mass = mujoco.mj_getTotalmass(self.model)
        print(f"시스템 준비 완료: 로봇 질량 {self.total_mass:.2f} kg 감지됨.")

        # 힘을 가할 기준점(골반/Pelvis) ID 찾기
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_id == -1:
            self.pelvis_id = 1  # 못 찾으면 베이스 링크로 설정

        # 초기 자세 리셋 (Stand Pose)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def apply_levitation(self):
        """
        [PID 제어기 핵심]
        로봇을 공중에 안정적으로 띄우기 위한 제어 로직입니다.
        """
        target_height = 0.85  # 목표 높이 (다리가 땅에 안 닿게 설정)

        # 현재 상태 측정 (z좌표, 수직속도)
        # qpos[2]: Z축 위치, qvel[2]: Z축 속도
        current_z = self.data.qpos[2]
        current_vz = self.data.qvel[2]

        # 1. 중력 보상 (Gravity Compensation)
        # 로봇 무게(mg) 만큼 위로 힘을 주어 중력을 상쇄시킵니다.
        gravity_force = self.total_mass * 9.81

        # 2. PD 제어 (위치 유지)
        # - P gain (500): 목표 높이로 당기는 스프링 힘
        # - D gain (50): 흔들림을 잡아주는 댐퍼 역할 (튀는 현상 방지)
        kp = 500.0
        kd = 50.0
        control_force = kp * (target_height - current_z) - kd * current_vz

        # 최종 힘 적용 (골반을 위로 들어올림)
        self.data.xfrc_applied[self.pelvis_id, 2] = gravity_force + control_force

        # 3. 자세 안정화 (회전 댐핑)
        # 로봇이 공중에서 뱅글뱅글 도는 것을 막기 위해 회전 저항을 줍니다.
        # qvel[3:6]은 각속도(Angular Velocity)를 의미합니다.
        self.data.xfrc_applied[self.pelvis_id, 3:6] = -20.0 * self.data.qvel[3:6]

    def run(self):
        # 시뮬레이션 뷰어 실행
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("=== G1 Air-Walking 데모 시작 ===")
            print("현재 골반을 공중에 고정(PID Control)하고 보행 알고리즘을 검증 중입니다.")

            while viewer.is_running():
                step_start = time.time()
                t = self.data.time

                # 1. 공중부양 제어 실행 (넘어짐 방지)
                self.apply_levitation()

                # 2. 보행 궤적 계산
                l_target, r_target = self.gait.get_joint_targets(t)

                # 3. 역기구학(IK) 풀이
                l_angles = self.kin.leg_IK(*l_target)
                r_angles = self.kin.leg_IK(*r_target)

                # 4. 모터에 제어 신호 입력
                # 왼쪽 다리 (Actuator 0~5)
                self.data.ctrl[0:6] = l_angles
                # 오른쪽 다리 (Actuator 6~11)
                self.data.ctrl[6:12] = r_angles

                # 5. 상체 자세 고정 (안정성 확보)
                # 허리(Waist) 고정
                self.data.ctrl[14] = 0.0
                # 팔(Arm) 고정 (초기 자세 유지)
                if self.model.nkey > 0:
                    self.data.ctrl[15:29] = self.model.key_ctrl[0, 15:29]

                # 물리 엔진 업데이트
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 시뮬레이션 속도 동기화
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)


if __name__ == "__main__":
    # XML 파일 경로 설정 (상황에 맞게 자동 탐색)
    xml_path = "scene.xml"
    if not os.path.exists(xml_path):
        xml_path = "../description/scene.xml"

    sim = G1LevitationSim(xml_path)
    sim.run()
```

# 참고 사이트
https://github.com/unitreerobotics/unitree_rl_gym
https://github.com/unitreerobotics/unitree_mujoco/tree/main
