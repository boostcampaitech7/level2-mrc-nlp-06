import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from evaluate import load
import os, ast, json, argparse
from datasets import load_from_disk

"""
DATA_PATH: corpus의 개행이 답변에 포함되지 않도록 개행 처리된 버전의 데이터셋 입력
RETRIEVED_PATH: sparse retrieval을 수행 후 생성되는 csv 파일 경로 입력
"""
DATA_PATH = "/data/ephemeral/home/datasets/v0.0.2"
RETRIEVED_PATH = "../../../outputsbm25.csv"


def system_prompt(pred_type):
    system = ""
    if pred_type == "valid":
        system = """당신은 기계 독해를 수행하는 전문가입니다. 질문을 보고 함께 제공된 지문에서 가장 명확한 답변을 추출하세요.

[지시사항]
1. 질문이 입력되면 함께 제공된 지문으로부터 가장 명확한 답변을 찾으세요.
2. 답변을 찾을 때:
  - 답변이 여러 개인 경우, 질문과 지문의 문맥을 고려하여 가장 유사한 답변을 고르세요.
  - 질문과 답변의 맥락을 고려했을 때 답변에 불필요하게 중복되는 단어가 없어야 합니다.
  - 질문에 순서 정보가 명시되어 있을 경우 순서 또는 인과관계를 파악하여 답변을 추출해야 합니다.

[제한사항]
- 질문과 지문의 맥락을 파악하여 답변을 추출하세요.
- 추출한 답변은 반드시 지문에 명시된 글자들로만 구성되어야 합니다.
- 고유명사는 형태를 그대로 유지하세요.
- 추출한 답변에는 따옴표, 괄호와 같은 특수문자가 포함될 수 있습니다.
- 문장 형태의 답변을 피하고, 짧은 구나 구절을 선택하세요."""

    elif pred_type == "test":
        system = """당신은 기계 독해를 수행하는 전문가입니다. 질문을 보고 함께 제공된 지문에서 가장 명확한 답변을 추출하세요.

[지시사항]
1. 질문이 입력되면 개행(\n)으로 구분된 서로 다른 지문들 중 답변이 있을 만한 지문을 한 개만 찾으세요.
  - 질문의 핵심을 정확히 확인하여 핵심이 포함된 지문을 선택하세요.
  - 질문이 모호하여 관련 지문이 없는 것처럼 보인다면, 가장 유사한 지문을 선택하세요.
2. 지문에서 답변을 찾으세요.
  - 답변이 여러 개거나 없는 경우, 질문과 지문의 문맥을 고려하여 가장 유사한 답변을 고르세요.
  - 질문과 답변의 맥락을 고려했을 때 답변에 불필요하게 중복되는 단어가 없어야 합니다.
  - 질문에 순서 정보가 명시되어 있을 경우 순서 또는 인과관계를 파악하여 답변을 추출해야 합니다.

[제한사항]
- 질문과 지문의 맥락을 파악하여 답변을 추출하세요.
- 추출한 답변은 반드시 지문에 명시된 글자들로만 구성되어야 합니다.
- 고유명사는 형태를 그대로 유지하세요.
- 추출한 답변에는 따옴표, 괄호와 같은 특수문자가 포함될 수 있습니다.
- 문장 형태의 답변을 피하고, 짧은 구나 구절을 선택하세요.
"""

    return system


def fewshot_prompt():
    fewshot = {
        "ex0": {
            "question": "테런스 엄마의 직업은?",
            "context": '1976년 2월 29일 미국 앨라배마주 몽고메리에서 태어나, 옆 동네 밀브룩에서 자랐다 그의 부모님은 롱이 아주 어렸을 때 이혼을 했다 어머니 낸시 앤 롱은 첫 아이를 16살 때 가졌고, 테런스를 낳았을 당시는 24살이었다 어머니가 학교를 졸업하고 집에서 2시간 거리의 헌츠빌에서 교도관으로 일하며 생계를 유지하는 동안, 할머니 소피아 피터슨이 롱과 그의 형들, 케이스와 하비에르 리처드를 키워냈다 롱은 후일 "내가 지금 여기 있을 수 있는 이유는 모두 할머니 덕분이라고 생각한다. … 마약과 갱들로 가득찬 거리에서 할머니는 우리를 보호해주셨다."라고 말했다 그의 할아버지 아델은 젊은 시절 그 지역 최고의 세미프로 투수였다고 한다 그는 다른 프로 야구 선수들보다 비교적 늦은 나이인 13살에 야구를 시작했다 그 이유는 롱 자신은 더 어릴 적부터 야구를 하고 싶어 했지만, 가난한 집안 사정이 그를 돕지 못했기 때문이었다 그는 "가족들이 내가 하는 이야기에 관한 거라곤 온통 야구 뿐이었다고 말해요. 하지만 나는 그렇게 하지 못했죠. 왜냐하면 그걸 감당할 수 없었으니까요. 중학생이 될 때까지 내가 야구에 관한 했던 것이라곤 테니스공과 막대기를 가지고 노는 것이었어요. 나는 13살이 되어서야 내 글러브를 처음 가져봤어요."라고 이야기했다 그가 야구를 시작하게 된 계기는 어느날 한 유스 리그 경기에서 코치가 선수들이 부족해 걱정하고 있었는데, 마침 경기장 주변을 배회하던 그에게 코치가 우연히 경기 출전을 권유하면서 부터라고 한다. 그는 그 경기에서 3이닝 동안 9삼진을 잡는 동시에 4안타를 쳤고, 그 이후 계속 야구를 하게 되었다고 한다 스탠호프 엘모어 고등학교에 진학한 이후 대학교 장학금을 받기 위해 농구에 집중하려 했지만, 고등학교 마지막 해에 22경기 15홈런 60타점, .608의 고타율을 기록하면서 스카우트들의 관심을 받았다 결국 1994년 메이저 리그 베이스볼 드래프트 1라운드(전체 20순위)에서 뉴욕 메츠의 두 번째 선수로 지명되었고, 그 후 약 2주 뒤 메츠와 50만 달러의 계약을 맺었다.',
            "answer": "교도관",
        },
        "ex1": {
            "question": "전쟁을 일으킨 당사자와 관련자들을 처벌하기 위해 거행된 재판이 열린 곳은?",
            "context": "포츠담 선언에 의해서 독일 본토는 연합국에 의해 분할 점령되었고 독일은 옛 영토의 3분의 1을 잃었다. 대부분은 폴란드 영토로 귀속되었고, 동프로이센의 반은 소련에 병합되었다. 체코슬로바키아, 헝가리, 루마니아, 유고슬라비아 등에서 오히려 소수 민족이 된 약 1,000만 명의 독일인들은 추방되었다. 미국·영국·프랑스의 서쪽 점령 지역은 독일연방공화국(서독)이 되었으며, 소련의 동쪽 점령 지역은 독일민주공화국(동독)이 되었다. 전쟁을 일으킨 장본인인 히틀러와 요제프 괴벨스, 하인리히 힘러는 패전을 전후로 자살하였으며, 카를 하우스호퍼, 헤르만 괴링, 빌헬름 카이텔, 요아힘 폰 리벤트로프, 루돌프 헤스, 카를 되니츠 등의 나머지 수뇌들은 연합군에 의한 전범 재판인 뉘른베르크 재판에서 판결을 받고 처형되거나 구금되었다. 그 외에도 히틀러가 고용한 영화 감독으로 알려진 레니 리펜슈탈이나, 나치 고관이었던 애인의 비호 아래 방탕한 생활을 보내고 있던 코코 샤넬, 나치의 적극적 협력한 지휘자 카라얀 등 나치 독일의 범죄 행위에 가담한 예술가나 실업가 등도 전후 국적을 불문하고 죄를 추궁당하여 대거 활동이 금지되었다. 전범 처벌은 유명인사들 뿐만 아니라 일반인에게도 행해졌으며 프랑스의 경우 나치 독일에 협력했던 사람들을 전부 잡아다 남자는 총살시키고 여자는 삭발 후 속옷 바람으로 조리돌림을 시킨 후 태형에 처했다. 모든 반파시스트 유럽 국가에서는 나치 및 파시스트의 구성원을 처벌하는 법률이 확립되었으며, 종전 전에 도주한 사람들도 국제 수배자 명단에 오르게 되어 결국은 체포되어 처벌되었다. 전후 독일에서는 나치당 출신으로 외무부에서 일했던 쿠르트 게오르크 키징어가 잠시 총리로 재직하기도 하였으나, 하인리히 뵐, 귄터 그라스 등 당시 지식인들의 비판이 끊이지 않았고, 결국 2년만에 총선에 패배하여 사민당에게 정권을 내주게 된다.",
            "answer": "뉘른베르크",
        },
        "ex2": {
            "question": "류한욱이 두 번째 뇌출혈로 쓰러진 공간은?",
            "context": "평안북도 철산군 부서면에서 소작농의 외아들로 태어났다. 어릴 때 어머니를 여의고 아버지와 함께 떠돌이로 어려운 생활을 하였다. 신의주에 정착한 뒤 학교를 다니면서부터 반일 감정을 키웠다. 이 지역은 압록강과 가까워 만주 지역의 항일 운동 소식이 자주 들려왔으며, 일본 군경의 활동도 활발하였기 때문이다. 신의주에서부터 축구 등 스포츠에 뛰어났던 류한욱은 만주로 건너가 펑톈에서 권투 선수가 되었다. 일제 강점기 동안 권투 선수로 활동하다가 태평양 전쟁이 끝나자 고향 철산으로 돌아왔다. 북조선로동당에 입당하고 철산군 백량면 면당 위원장을 맡아 북조선임시인민위원회의 토지개혁 정책을 시행하였다. 평북경찰서에서 검사를 지내기도 했다. 1954년 가을에 공작원으로 남파되었다가 의정부에서 총상을 입고 체포되었다. 전향을 거부하고 비전향 장기수로 수감 생활을 하던 중, 1969년에 대전교도소에서 뇌출혈로 쓰러졌다. 운동선수 출신으로 건강했던 류한욱은 자신이 뇌출혈을 일으킨 것이 고문에 의한 것이라고 주장하였다. 이때의 뇌출혈로 반신불수가 되었다. 이후 광주교도소에서 다시 한번 뇌출혈을 일으켜 몸을 움직이지 못하고 말도 하지 못하게 되었다. 음식은 감방 동료가 입에 넣어주어야 먹을 수 있었다. 류한욱은 이때도 운동 중에 간수에게 구타당하면서 뇌출혈을 일으켰다고 밝혔다. 전향을 하지 않아 치료를 받지 못하다가, 1991년 2월 22일에 회생 가능성이 없는 환자로 분류되어 출감하였다. 총 수감 기간은 약 36년이다. 충청북도 음성군의 꽃동네를 거쳐 서울 관악구에서 다른 출소 장기수들과 공동생활을 하였다. 비전향 장기수를 소재로 한 다큐멘터리 영화 《송환》 촬영에도 응하였다. 영화 속의 류한욱은 몸이 불편해 거동하지 못하면서도 강인한 기상을 잃지 않는 인물로 그려졌다. 2000년 6·15 남북 공동선언에 의해 조선민주주의인민공화국으로 송환되고 조국통일상을 받았다.",
            "answer": "광주교도소",
        },
        "ex3": {
            "question": "병에 걸려 죽을 확률이 약 25~50%에 달하는 유형의 질병은?",
            "context": "보통 유형 천연두의 치사율은 약 30%지만, 농포 분포에 따라 달라질 수 있었다. 보통 유형 융합성 천연두의 치사율은 50 ~ 75%이고, 보통 유형 준융합성 천연두는 약 25 ~ 50%였다. 발진이 이산적일 경우 치사율은 10% 이하였다. 1세 이하 영아의 치사율은 유형을 막론하고 40 ~ 50%이다. 출혈성 및 악성 천연두는 치사율이 매우 높았다. 악성 천연두의 치사율은 90% 이상이었고 출혈성 천연두의 치사율은 거의 100%였다. 소두창의 치사율은 1% 이하였다 천연두바이러스가 만성적이거나 재발할 수 있다는 증거는 없다 보통 유형 천연두가 치명적일 경우, 대개 감염 10일 ~ 16일차에 사망한다. 천연두로 인한 사망원인은 명확하지 않으나, 그 감염이 다수의 장기와 관련되어 있음은 밝혀졌다. 면역복합체 순환, 압도적 바이러스혈증, 통제불능의 면역반응 등이 죽음에 기여하는 요소들일 수 있다 초기 출혈성 천연두의 경우 발열 이후 약 6일 정도에 갑자기 죽는다. 출혈성 천연두의 사망원인은 심부종이며, 때때로 폐부종이 수반될 수 있다. 말기 출혈성 천연두의 경우 지속적 바이러스혈증, 심각한 혈소판 감소, 면역반응의 무력화 등이 사망원인으로 거론되었다 악성 천연두의 사망원인은 체액·단백질·전해액이 생명유지를 위해 필요한 양에 미달하거나, 전격성 패혈증이 일어나서 등 화상과 유사하다.",
            "answer": "보통 유형 준융합성 천연두",
        },
        "ex4": {
            "question": "소전제와 대전제에서 나타나는 제3의 개념을 무엇이라고 부르는가?",
            "context": "삼단논법(三段論法)은 미리 알려진 두 판단에서 그것들과는 다른 하나의 새로운 판단으로 이끄는 추론 방법이다. 2개의 명제를 전제로 결론을 내는 대표적인 간접추론 형식이자 연역추론이다. 모든 사람은 죽는다. 소크라테스는 사람이다. 그러므로 소크라테스는 죽는다. 같은 추리가 대표적인 것이다. 결론에서 주어 '인간'을 소개념, 술어 '죽어야만 하는 것'을 대개념이라 하고, 소개념을 포함한 전제를 소전제(小前提), 대개념을 포함한 전제를 대전제(大前提)라 한다. 두 전제에는 대소개념과는 다른 제3의 개념 '동물'이 포함되어 있다. 이는 두 전제를 결부시켜 결론으로 이끌기 위한 매개적 작용을 나타내는 것으로서 매개념(媒槪念)이라고 한다. 일반화하자면, 대전제는 결론의 술어 개념인 대개념을 포함한 전제이고, 소전제는 결론의 주어 개념인 소개념을 포함한 전제이며, 매개념은 두 전제에서만 나타나며 결론에서는 나타나지 않는다. 소개념을 S, 매개념을 M, 대개념을 D로 나타내는 것이 보통이다. 표준형식삼단논법에서는 대전제가 먼저 진술되고 그 다음에 소전제가 진술된다. 그러나 대전제와 소전제는 위치에 따라 정해지는 것이 아니라 대개념과 소개념의 포함 여부로 결정된다.",
            "answer": "매개념(媒槪念)",
        },
        "ex5": {
            "question": "라쉬를 반여성주의자라고 칭했던 인물은?",
            "context": "1980년대에 래시는 현대 미국의 주요 정치적인 사상에 대해 경멸하게 되고 이것에 대해 분노한 자유주의자들은 진보주의와 여성주의를 비판하였다. 그는 “과거의 여성의 업적을 존중하는 여성주의운동이라면 가사, 어머니의 역할, 이웃사회를 위한 봉사를 폄하하지 않을 것이다”라고 언급했다. 성취의 상징이 급료만 될 수 있게 하지 않을 것이다. 그는 사람들이 월급은 높지만 가족과의 시간을 뺏는 화려한 직업보다는 자존감을 지킬 수 있는 명예로운 직업이 필요하다고 주장했다. 진보 기자인 Susan Faludi는 라쉬가 낙태할 수 있는 권리에 대한 운동을 비판하고 이혼에 반대했기 때문에 반여성주의자라고 불렀다. 그러나 래시는 Ronald Reagan의 보수주의를 전통과 도덕적인 책임의 대조라고 여겼다. 래시는 일반적으로 그때 당시의 뉴라이트(New Right)의 원인, 특히 자유의지론의 요소들에 대해 동조하지 않았고, 미국사회의 모든 면에 자본주의가 침범하는 점을 몹시 싫어했다. 래시는 사회적 용인과 경제의 중앙집중이 미국의 진보적인 이상의 기반을 형성한 뉴딜 정책이 생겨날 때쯤에 출현한 지배적인 정치 기라성을 거부함과 동시에 William F. Buckley 와 Russell Kirk에 의해 만들어진 진보주의와는 전혀 다른 종합적인 보수주의에 대해서도 질책했다. 래시는 그의 사회철학과 가장 가까운 사상인 사회주의에 대해서도 놀랍게도 비판적이었고 때로는 무시하기도 하였다. 오직 포퓰리즘만이 경제적 공정성 (꼭 평등이 아니더라도 계급을 줄이는 것), 참여 민주주주의, 사회결합 그리고 도덕적 준엄에 대한 래시의 기준에 부합한다. 그러나 포퓰리즘은 뉴딜 정책 시기에 중대한 실수를 범하게 되고 갈수록 적들을 끌어들이게 되고 동맹들에게 무시 받게 된다. 예를 들면 그는 마틴 루터 킹의 초기 사상을 미국 포퓰리즘의 본보기로 여겼다. 그러나 래시의 관점에서 마틴 루터 킹은 말년에 진행 중이었던 인종의 계층화에 대해 근본적으로 관료주의적인 해답을 받아드리면서 그의 급진적인 비전에 미치지 못하였다.",
            "answer": "Susan Faludi",
        },
        "ex6": {
            "question": "동양의 문화적 요소를 가장 잘 활용한 타란티노의 작품은?",
            "context": '타란티노는 이야기의 비선형 구성을 자주 구사한다. 그가 감독한 거의 모든 영화에선 시간대로, 순차적으로 이야기가 진행된 경우가 없을 정도로 복잡한 시간 배열과, 두 가지 이상의 이야기를 엮어 놓는 다중 플롯 등을 기본 뼈대로 주로 사용하였다. 그의 영화에는 대사량이 무척 많기로 유명하다. 특별히 사건의 핵심이 되는 대사 외에 시시껄렁한 농담이나 조롱 등의 대화가 넘쳐날 정도로 많은 것이 타란티노의 전매 특허이다. 수다가 곧 영화의 핵심이라 할 정도이다. 영화 음악으로는 별도의 음악 감독과 함께 작업하여 순수 창작곡을 사용하는 것이 아닌, 타란티노 본인이 직접 고전 팝 음악을 선곡하여 영화 속에 삽입하는 식으로 영화 음악을 작업한다. B급 영화의 요소들을 선호하며, 이것을 주류 영화로 재창조하는 데에 선구자적인 역할을 하였다. B급 영화 중 기괴하고, 폭력이 난무하고, 과장된 성적 표현 등의 요소들을 현대적으로 새롭게 버무리는 데에 능숙하다. 고전 필름 누아르, 스파게티 웨스턴 등의 장르 역시도 타란티노의 탁월한 현대적 재창조 대상에 포함되었다. 그의 이러한 성향은 커다란 파급 효과를 나타냈으며, 1990년대부터 세계적으로 이와 비슷한 유형의 영화들이 많이 만들어지기도 했다. 가이 리치 (Guy Ritchie), 에드거 라이트 (Edgar Wright), 그리고 류승완 등 "제2의 타란티노"란 수식어가 붙은 신예 감독들이 대거 등장하기도 했다. 동양 문화 중 일본의 사무라이, 중국의 무협 등을 동경하며 그것을 서구적으로 재창조하기도 했다. 가장 두드러지게 표현한 영화는 《킬 빌》이며, 이 영화에서는 이러한 모든 요소들이 오마주로 표현되기도 하였다. 그는 1960년대 홍콩 쇼 브라더스 영화들과 70년대의 이소룡 영화, 그리고 고전 사무라이 영화들을 광적으로 좋아하기로 유명하다. 그에 대한 평가는 자국인 미국보다 프랑스 등의 유럽에서 더 좋은 경향이 있다. 미국에서 가장 큰 규모의 영화상인 아카데미 상과 골든 글러브 상에선 상복이 별로 없는 편이지만, 대신 프랑스 칸 영화제에서 《펄프 픽션》을 통해 황금 종려상을 받았고, 그 외에도 꾸준히 유럽의 여러 영화제에 작품을 출품하거나, 심사위원으로 참가하는 등 활발한 모습을 보여 주었다. 영화 창작에 있어 로버트 로드리게스와 공동 작업을 많이 하며, 배우로는 새뮤엘 L. 잭슨, 우마 써먼, 마이클 매드슨, 브루스 윌리스, 하비 키이텔, 크리스토프 발츠 등을 주로 캐스팅한다.',
            "answer": "《킬 빌》",
        },
        "ex7": {
            "question": "데메카론에는 무엇을 풍자하는 이야기가 들어있나요?",
            "context": """이야기에서 성 프란체스코 수도회의 수도사 라고만 언급되나, 조반니 빌라니의 연대기와 대조해 보면, 그 이름이 피에트로 달라키임을 확인할 수 있다는 것이 알려져 있다. 피에트로 달라키는 재물을 밝히는 수도사로, 어느 부자가 자기 집에 있는 포도주를 자랑하면서 술김에 "예수 그리스도께서 마실 만한 포도주"라는 표현을 썼다는 점을 트집 잡아서 돈을 뜯어내려고 한다. 피에트로 달라키는 이 부자가 "그리스도가 술 주정뱅이라는 식으로 신성모독 표현을 사용했다"고 하여, 종교 재판으로 화형에 처해버리려고 한다. 부자는 살려달라고 하면서 뇌물을 바치고, 피에트로 달라키는 많은 돈을 받고, 매일 수도원에서 경건히 기도하게 하는 조건으로 화형을 면해 준다. 나중에 기도 생활의 소감을 한 번 말해 보라고 하자, 부자는 "매일 수도원에 수프가 남아도는 것을 보았는데, '하나에 대해 백을 받게 될 것이다'라는 말이 있으니, 수도원 사람들은 지옥에서 수프의 바다에 빠져 죽지 않을까 걱정이다"라고 말한다. 수도사는 부자를 꺼림칙하게 여겨, 기도를 멈추고 집에 그냥 돌아가게 한다. 데카메론에서 언급된 이야기 중에는 "돈은 욕심 많은 성직자의 악질 탐욕병에는 매우 큰 효과가 있는 법이다", "이 미약은 그 효과가 비할 것이 없어서, 갈레노스의 의학서에는 써 있지 않습니다만, 그 효험 덕분에 화형을 십자가로 바뀌었다"라는 식으로 뇌물을 풍자하는 표현이 등장한다.""",
            "answer": "뇌물",
        },
        "ex8": {
            "question": "다른 과 의사들은 감염내과 전문의들로부터 어떤 것에 대해 조언을 받는가?",
            "context": "감염내과 전문의들은 일반적으로 다른 진료과의 의사들에게 복잡한 감염병에 대한 고문 역할을 하며, 종종 인간면역결핍 바이러스/후천면역결핍증후군과 같은 면역 결핍 환자들을 직접 관리하기도 한다. 대부분의 감염병들이 특별히 감염내과 전문의의 진료 및 치료를 필요로 하지는 않지만, 증상의 모호함 등으로 인해 감염병 환자를 진단하거나 관리하기 어려운 경우에는 감염내과 전문의들의 조언을 받아 진료가 이루어진다. 열병의 감염원을 밝히기 위한 조언을 해줄수도 있다 감염내과 전문의들은 또한 병원(hospital, 입원환자)과 진료소(clinic, 외래환자)에서 모두 근무할 수 있다. 병원에서는 감염내과 전문의가 감염원을 파악하기 위해 적합한 진단검사를 제안하여 적기에 진단될 수 있게 하며, 세균 감염을 치료하기 위해 항생제 처방 등 적절한 치료법을 제안한다. 실제로 몇몇 감염병의 경우 감염내과 의사가 치료에 참여함으로써 환자의 예후가 상당히 좋아진다. 감염내과 전문의들은 진료소에서는 후천면역결핍증후군 등 만성 감염 환자에게 장기적 치료를 제공할 수 있다.",
            "answer": "복잡한 감염병",
        },
    }

    for item in fewshot.items():
        fewshot[item[0]] = {
            "user": "\n".join(
                ["질문: " + item[1]["question"], "지문: " + item[1]["context"]]
            ),
            "assistant": item[1]["answer"],
        }

    fewshot_mini = "\n\n".join(
        [
            f"예시{idx+1}:\n{fewshot[key]['user']}\n답변: {fewshot[key]['assistant']}"
            for idx, key in enumerate(list(fewshot.keys()))
        ]
    )
    return fewshot_mini


def user_prompt(question, context):
    user = """질문: {question},
지문: {context},
답변: """

    user_filled = user.format(question=question, context=context)
    return user_filled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", type=str, default="valid", help="choose 'test' or 'valid' as default"
    )
    parser.add_argument("--key", type=str, help="GPT API key")
    args = parser.parse_args()

    client = OpenAI(api_key=args.key)

    results_dict = {}
    if args.type == "valid":
        system = system_prompt(pred_type="valid")
        fewshot = fewshot_prompt()

        valid_datasets = load_from_disk(DATA_PATH)["validation"]
        valid_datasets = pd.DataFrame(valid_datasets)

        for val_id in tqdm(
            list(valid_datasets["id"]), desc="validation", total=len(valid_datasets)
        ):
            question = list(
                valid_datasets.loc[valid_datasets["id"] == val_id, "question"]
            )[0]
            context = list(
                valid_datasets.loc[valid_datasets["id"] == val_id, "context"]
            )[0]

            user = user_prompt(question, context)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": fewshot},
                    {
                        "role": "assistant",
                        "content": "이해했습니다. 주어진 예시들을 참고하여 지문으로부터 가장 명확한 답변을 찾아내겠습니다.",
                    },
                    {"role": "user", "content": user},
                ],
            )

            result = response.choices[0].message.content
            results_dict[val_id] = result

        total = 0
        val_predicted = []
        ground_truth = []
        for val_id in list(valid_datasets["id"]):
            val_answers = list(
                valid_datasets.loc[valid_datasets["id"] == val_id, "answers"]
            )[0]
            ground_truth.append({"id": val_id, "answers": val_answers})

            pred_answers = results_dict[val_id]
            val_predicted.append({"id": val_id, "prediction_text": pred_answers})

        metric = load("squad")
        scores = metric.compute(predictions=val_predicted, references=ground_truth)
        print("Validation Prediction Result")
        print(scores)

    elif args.type == "test":
        with open(
            os.path.join(DATA_PATH, "wikipedia_documents.json"), "r", encoding="utf-8"
        ) as f:
            corpus = json.load(f)

        # Sparse Retrieval 후 생성되는 outputsbm25.csv 파일이 필요
        test_retrieved = pd.read_csv(RETRIEVED_PATH)
        test_ids = list(test_retrieved["id"])

        test_retrieved_corpus = []
        for id_ in test_ids:
            doc_id = ast.literal_eval(
                list(test_retrieved.loc[test_retrieved["id"] == id_, "document_id"])[0]
            )

            contexts = []
            for d_id in doc_id:
                contexts.append(corpus[str(d_id)]["text"])

            test_retrieved_corpus.append(
                {
                    "id": id_,
                    "question": list(
                        test_retrieved.loc[test_retrieved["id"] == id_, "question"]
                    )[0],
                    "context": contexts,
                }
            )

        system = system_prompt(pred_type="test")
        fewshot = fewshot_prompt()

        for data in tqdm(
            test_retrieved_corpus, desc="test", total=len(test_retrieved_corpus)
        ):
            t_id = data["id"]
            question = data["question"]
            context = "\n".join(data["context"])

            user = user_prompt(question=question, context=context)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 알아서 가장 최신 버전을 불러옴 -> gpt-4o-mini-2024-07-18
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": fewshot},
                    {
                        "role": "assistant",
                        "content": "이해했습니다. 주어진 예시들을 참고하여 지문을 찾고, 가장 명확한 답변을 찾아내겠습니다.",
                    },
                    {"role": "user", "content": user},
                ],
            )

            result = response.choices[0].message.content
            if "답변: " in result:
                result = result.split("답변: ")[-1]
            results_dict[t_id] = result
            print(results_dict)
            break

    with open(f"gpt_{args.type}_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)
