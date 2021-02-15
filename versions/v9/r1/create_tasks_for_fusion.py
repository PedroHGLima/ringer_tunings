
import os
basepath = os.getcwd()

path = basepath + '/Zee/v9/r1'


# Path into the caloba cluster
path_to_rings  = '/home/jodafons/tasks/Zee/v9_rg/r1/'
path_to_rings += 'user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97.v9_rg_et{ET}_eta{ETA}.r1'
path_to_shower = '/home/jodafons/tasks/Zee/v9_ss/r1/'
path_to_shower+= 'user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97.v9_ss_et{ET}_eta{ETA}.r1'



command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97.v9.et{ET}_eta{ETA}.r1 \
  -c user.jodafons.job_config.Zee_v9.10sorts.10inits.r1 \
  -d user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et{ET}_eta{ETA}.npz \
  --sd "{REF}" \
  --exec "run_tuning.py -c %IN -d %DATA -r %REF -v %OUT -t v9 -p r1 -b zee \
  --extraArgs '--type fusion --path_to_rings {RINGS} --path_to_shower {SHOWER}'" \
  --queue "gpu" """


try:
    os.makedirs(path)
except:
    pass



for et in range(5):
    for eta in range(5):
        ref = "{'%%REF':'user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et%d_eta%d.ref.pic.gz'}"%(et,eta)
        cmd = command.format(PATH=path, ET=et,ETA=eta,REF=ref, RINGS=path_to_rings.format(ET=et,ETA=eta), SHOWER=path_to_shower.format(ET=et,ETA=eta) )
        print(cmd)
        os.system(cmd)


