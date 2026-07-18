from agents.bc import BCAgent
from agents.crl import CRLAgent
from agents.dads import DADSAgent
from agents.ddpgbc import DDPGBCAgent
from agents.dds import DDSAgent
from agents.empowerment_crl import EmpowermentCRLAgent
from agents.empowerment_mine import EmpowermentMineAgent
from agents.empowerment_skill import EmpowermentAgent as EmpowermentSkillAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.opal import OPALAgent
from agents.qrl import QRLAgent
from agents.quest import QueSTAgent
from agents.sac import SACAgent
from agents.skill_dt import SkillDTAgent
from agents.skill_match import SkillMatchAgent
from agents.vq_bet import VQBeTAgent

agents = dict(
    bc=BCAgent,
    crl=CRLAgent,
    dads=DADSAgent,
    ddpgbc=DDPGBCAgent,
    dds=DDSAgent,
    empowerment_crl=EmpowermentCRLAgent,
    empowerment_mine=EmpowermentMineAgent,
    empowerment_skill=EmpowermentSkillAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    opal=OPALAgent,
    qrl=QRLAgent,
    quest=QueSTAgent,
    sac=SACAgent,
    skill_dt=SkillDTAgent,
    skill_match=SkillMatchAgent,
    vq_bet=VQBeTAgent,
)
