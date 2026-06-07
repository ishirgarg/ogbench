from agents.bc import BCAgent
from agents.crl import CRLAgent
from agents.dads import DADSAgent
from agents.ddpgbc import DDPGBCAgent
from agents.empowerment_crl import EmpowermentCRLAgent
from agents.empowerment_mine import EmpowermentMineAgent
from agents.empowerment_skill import EmpowermentAgent as EmpowermentSkillAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.skill_match import SkillMatchAgent

agents = dict(
    bc=BCAgent,
    crl=CRLAgent,
    dads=DADSAgent,
    ddpgbc=DDPGBCAgent,
    empowerment_crl=EmpowermentCRLAgent,
    empowerment_mine=EmpowermentMineAgent,
    empowerment_skill=EmpowermentSkillAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    skill_match=SkillMatchAgent,
)
