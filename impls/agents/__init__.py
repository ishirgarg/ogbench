from agents.bc import BCAgent
from agents.crl import CRLAgent
from agents.ddpgbc import DDPGBCAgent
from agents.empowerment_skill import EmpowermentAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent

agents = dict(
    bc=BCAgent,
    crl=CRLAgent,
    ddpgbc=DDPGBCAgent,
    empowerment_skill=EmpowermentAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)
