import os
basepath = os.getcwd()


path = basepath + '/jpsi/v1/r1'


command = """maestro.py task create \
  -v {PATH} \
  -t user.mverissi.data17-18_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97.v1_et{ET}_eta{ETA}.r1 \
  -c user.mverissi.job_config.Jpsi_v1.n2to5.10sorts.5inits.r1 \
  -d user.mverissi.data17-18_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et{ET}_eta{ETA}.npz \
  --sd "{REF}" \
  --exec "run_tuning.py -c %IN -d %DATA -r %REF -v %OUT -t v1 -b jpsi -p r1" \
  --queue "gpu" """

try:
    os.makedirs(path)
except:
    pass

for et in range(3):
    for eta in range(5):
        ref = "{'%%REF':'user.mverissi.data17-18_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et%d_eta%d.ref.pic.gz'}"%(et,eta)
        cmd = command.format(ET=et,ETA=eta,REF=ref, PATH=path)
        print(cmd)
        os.system(cmd)