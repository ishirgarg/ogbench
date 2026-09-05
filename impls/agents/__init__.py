from agents.bc import BCAgent
from agents.crl import CRLAgent
from agents.dads import DADSAgent
from agents.ddpgbc import DDPGBCAgent
from agents.dds import DDSAgent
from agents.dds_controller import DDSControllerAgent
from agents.empowerment_crl import EmpowermentCRLAgent
from agents.empowerment_crl_flowbc import EmpowermentCRLFlowBCAgent
from agents.empowerment_dads import EmpowermentDADSAgent
from agents.empowerment_dv import EmpowermentDVAgent
from agents.empowerment_skill import EmpowermentAgent as EmpowermentSkillAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.online_crl import OnlineCRLAgent
from agents.online_crl_skill_controller import OnlineCRLSkillControllerAgent
from agents.opal import OPALAgent
from agents.opal_controller import OPALControllerAgent
from agents.qrl import QRLAgent
from agents.quest import QueSTAgent
from agents.sac import SACAgent
from agents.skill_bc_relabel_controller import SkillBCRelabelControllerAgent
from agents.skill_dt import SkillDTAgent
from agents.skill_dt_controller import SkillDTControllerAgent
from agents.skill_match import SkillMatchAgent
from agents.skill_value_controller import SkillValueControllerAgent
from agents.vq_bet import VQBeTAgent

agents = dict(
    bc=BCAgent,
    crl=CRLAgent,
    dads=DADSAgent,
    ddpgbc=DDPGBCAgent,
    dds=DDSAgent,
    dds_controller=DDSControllerAgent,
    empowerment_crl=EmpowermentCRLAgent,
    empowerment_crl_flowbc=EmpowermentCRLFlowBCAgent,
    empowerment_dads=EmpowermentDADSAgent,
    empowerment_dv=EmpowermentDVAgent,
    empowerment_skill=EmpowermentSkillAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    online_crl=OnlineCRLAgent,
    online_crl_skill_controller=OnlineCRLSkillControllerAgent,
    opal=OPALAgent,
    opal_controller=OPALControllerAgent,
    qrl=QRLAgent,
    quest=QueSTAgent,
    sac=SACAgent,
    skill_bc_relabel_controller=SkillBCRelabelControllerAgent,
    skill_dt=SkillDTAgent,
    skill_dt_controller=SkillDTControllerAgent,
    skill_match=SkillMatchAgent,
    skill_value_controller=SkillValueControllerAgent,
    vq_bet=VQBeTAgent,
)
