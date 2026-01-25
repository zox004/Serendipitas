import rlcard
from agents.random_agent import TexasRandomAgent

def run_day1_analysis():
    # 1. 환경 생성 (No-Limit Texas Hold'em)
    env = rlcard.make("no-limit-holdem", config={'allow_step_back': True})
    
    # 2. 에이전트 설정 (2인 플레이)
    agent = TexasRandomAgent(num_actions=env.num_actions)
    env.set_agents([agent, agent])

    print(f"변수 정보:")
    print(f"- 액션 개수: {env.num_actions}")
    print(f"- 에이전트 수: {env.num_players}")

    # 3. 게임 1회 실행 및 데이터 구조 분석
    trajectories, payoffs = env.run(is_training=False)
    
    # 첫 번째 스텝의 관측치 데이터 출력
    first_state = trajectories[0][0]
    print("\n--- State 데이터 구조 분석 ---")
    print(f"Key 목록: {first_state.keys()}")
    print(f"Obs 벡터 형태: {first_state['obs'].shape}")
    print(f"가능한 액션: {first_state['legal_actions']}")
    print(f"Raw 관측치(예시): {first_state['raw_obs']['hand']}") # 내 손패 확인
    
    print("\n--- 게임 결과 ---")
    print(f"최종 수익(Payoffs): {payoffs}")

if __name__ == "__main__":
    run_day1_analysis()