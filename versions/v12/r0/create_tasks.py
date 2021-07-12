import os
basepath = os.getcwd()

path = basepath + '/Zee/v8/r0'

command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97.v8_et{ET}_eta{ETA}.r0 \
  -c user.jodafons.job_config.Zee_v8.n2to5.10sorts.10inits.r0 \
  -d user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et{ET}_eta{ETA}.npz \
  --sd "{REF}" \
  --exec "run_tuning.py -c %IN -d %DATA -r %REF -v %OUT -b zee -t v8 -p r0" \
  --queue "gpu" """

try:
    os.makedirs(path)
except:
    pass

for et in range(5):
    for eta in range(5):
        ref = "{'%%REF':'user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et%d_eta%d.ref.pic.gz'}"%(et,eta)
        cmd = command.format(ET=et,ETA=eta,REF=ref,PATH=path)
        print(cmd)
        os.system(cmd)


