from quantifiers.CC import classify_count
from quantifiers.ACC import ACC
from quantifiers.PCC import PCC
from quantifiers.PACC import PACC
from quantifiers.HDy import Hdy
from quantifiers.X import X
from quantifiers.MAX import Max
from quantifiers.SMM import SMM
from quantifiers.dys_method import dys_method
from quantifiers.sord import SORD_method
from quantifiers.MS import MS_method
from quantifiers.MS_2 import MS_method2
from quantifiers.T50 import T50
from quantifiers.QT import QT
from quantifiers.QT_ACC import QT_ACC
from quantifiers.GAC import GAC
from quantifiers.GPAC import GPAC
from quantifiers.FM import FM
from utils.predictQuantifierSchumacherGithub import predictQuantifierSchumacherGithub

methods = {
  "CC": {
    "func": classify_count,
    "kargs": {},
  },
  
  "ACC": {
    "func": ACC,
    "kargs": {},
  },
  "PCC": {
    "func": PCC,
    "kargs": {},
  },

  "PACC": {
    "func": PACC,
    "kargs": {},
  },
  "HDy": {
    "func": Hdy,
    "kargs": {},
  },
  "MS": {
    "func": MS_method,
    "kargs": {}
  },

  "MS2": {
    "func": MS_method2,
    "kargs": {}
  },
  "X": {
    "func": X,
    "kargs": {}
  },
  "MAX": {
    "func": Max,
    "kargs": {}
  },
  "T50": {
    "func": T50 ,
    "kargs": {}
  },
  "SMM": {
    "func": SMM,
    "kargs": {}
  },

  "DyS": {
    "func": dys_method,
    "kargs": {}
  },
  "SORD": {
    "func": SORD_method,
    "kargs": {}
  },
  #"emq": {
   # "func": EMQ,
    #"kargs": {}
  #},
  "EMQ": {
    "func": {},
    "kargs": {}
  },
  "QT": {
    "func": QT,
    "kargs": {}
  },
  "PWK": {
    "func": {},
    "kargs": {}
  },
  "QT_ACC": {
    "func": QT_ACC,
    "kargs": {}
  },
  "Readme": {
    "func": predictQuantifierSchumacherGithub,
    "kargs": {}
  },
  "FM": {
    "func": predictQuantifierSchumacherGithub,
    "kargs": {}
  },
  "CDE": {
    "func": predictQuantifierSchumacherGithub,
    "kargs": {}
  },
  "HDx": {
    "func": predictQuantifierSchumacherGithub,
    "kargs": {}
  },
  "GPAC": {
    "func": predictQuantifierSchumacherGithub,
    "kargs": {}
  },
  "GAC": {
    "func": predictQuantifierSchumacherGithub,
    "kargs": {}
  },
  "FormanMM": {
    "func": predictQuantifierSchumacherGithub,
    "kargs": {}
  },
  "EM": {
    "func": predictQuantifierSchumacherGithub,
    "kargs": {}
  },
  
  #"GAC": {
   # "func": GAC,
    #"kargs": {}
  #},
  #"GPAC": {
   # "func": GPAC,
    #"kargs": {}
  #},
  #"FM": {
   # "func": FM,
    #"kargs": {}
  #},

}