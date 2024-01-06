import ROOT as rt
import numpy as np
import argparse as arg

parser = arg.ArgumentParser(description='Train Binary BDT')
parser.add_argument('-t', '--tag', dest='tag', type=str, default='', help="tag of folder used")
args = parser.parse_args()
tag = args.tag
print(tag)

#PU_000_Pt_025_Eta_2pt20

pt = tag.split('_')[3]
#print(pt)
arr = np.loadtxt('Energy_log_' + tag + '.txt')
h = rt.TH1F('h', 'Electron energy prediction at nPU=000 pt=' + pt + ' eta=2.2', 200, 0.5, 1.5) 
fn = rt.TF1("fn","gaus",0.7,1.4)
fn.SetNpx(1000)


for ar in arr:
    h.Fill(ar[1]/ar[0])

h.GetXaxis().SetTitle('Predicted Energy/True Energy')
rt.TGaxis.SetMaxDigits(3)
c1 = rt.TCanvas('c1', '', 600, 600)
c1.cd()
pad1 = rt.TPad('pad1', '', 0.0, 0.0, 1.0, 1.0)
pad1.Draw()
pad1.SetLeftMargin(0.12)
pave = rt.TPaveText(0.12, 0.5, 0.42, 0.9, 'NB,NDC')
pave.AddText('Window Size  : 0.5 ')
pave.AddText('Mip cut      : 0.5 ')
pave.AddText('Num min edges: 5   ')
pave.AddText('Epoch        : 2000')
pave.AddText('New Loss function (eval)')
h.SetLineWidth(5)
fn.SetLineWidth(5)
pad1.cd()
h.Draw('HIST')
fn.SetRange(h.GetMean()-3*h.GetRMS(),h.GetMean()+3*h.GetRMS())
fn.SetParameter(1,h.GetMean())
fn.SetParameter(2,h.GetRMS())
h.Fit('fn', 'LR')
fn.Draw('SAMES')
pave.Draw()
pad1.Update()
st = h.FindObject('stats')
st.Print()
st.SetOptStat(1110)
st.SetOptFit(101)
st.SetX1NDC(0.6)
st.SetX2NDC(0.9)
st.SetY1NDC(0.5)
st.SetY2NDC(0.9)

c1.Update()
c1.SaveAs('Electorn_energy_hist' + tag + '_2.png')
