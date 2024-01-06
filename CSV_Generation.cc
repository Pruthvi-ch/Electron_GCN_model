//g++ CSV_Generation.cc `root-config --cflags --libs` -lTMVA -lPhysics -o CSV_Generation.out
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLegend.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TF1.h"
#include "TString.h"
#include "TRandom3.h"
#include "TLine.h"
#include "TPaveText.h"
#include "THStack.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TSystem.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include <iostream>
#include <cmath>
#include <fstream>

float ADC_effective(int ADC, bool mode, int thick){
  float scale[5] = {0.0, 1.69, 1.0, 0.62, 2.3};
  float Eff_ADC = 0.0;
  if(mode){
    if(thick==4){
      Eff_ADC = scale[thick] * (37000.0 + (165.5 * ADC)) / 44.74;
    } else{
      Eff_ADC = scale[thick] * (1320 + (55.26 * ADC)) / 2.2;
    }
  } else{
    Eff_ADC = scale[thick] * ADC;
  }
  return Eff_ADC;
}


int CSV_Generation(){
  std::cout<< "start" << std::endl;
  //TFile* infile = new TFile("ele_new/Electorn_200.root", "READ");
  //TFile* infile = new TFile("/nfs/home/common/DataSet/Raw_Hits/Regular/Electron/Raw_Hits_Regular_Electron_PU_000_60k.root", "READ");
  TFile* infile = new TFile("/nfs/home/common/DataSet/Raw_Hits/Mono/Electron/Raw_Hits_Mono_Electron_PU_000_Pt_200_Eta_2pt20.root", "READ");
  // Z pisition fo all the layers
  double Z_[47] = {322.155,323.149,325.212,326.206,328.269,329.263,331.326,332.32,334.383,335.377,337.44,338.434,340.497,341.491,343.554,344.548,346.611,347.605,349.993,350.987,353.375,354.369,356.757,357.751,360.139,361.133,367.976,374.281,380.586,386.891,393.196,399.501,405.806,412.111,418.416,424.721,431.026,439.251,447.476,455.701,463.926,472.151,480.376,488.601,496.826,505.051,513.276};
  //Scaling of adc or energy
  double scale[] = {0, 1.69, 1, 0.63, 2.3};
  double scale_sim[] = {0, 1.69, 1, 0.63, 0.113};
  int in_id = 22;
  int n_layers = 47;
  int n = 1000; // number of events to analyze
  //TTree **intree = new TTree*[n_layers];
  TTree *intree = new TTree;
  //Variables for adc, nHit, x, y, simhitE, .etc 
  int nHit_ = 0;
  float X_[50000] = {0.0};
  float Y_[50000] = {0.0};
  float E_[50000] = {0.0};
  float t_[50000] = {0.0};
  uint16_t adc_[50000] = {0};
  UShort_t thick_[50000] = {0};
  uint16_t adc_mode_[50000] = {0};
  int16_t zside_[50000] = {0};
  
  int n_gen = 0;
  float eta_[100] = {0.0};
  float phi_[100] = {0.0};
  float pT_[100] = {0.0};
  float pz_[100] = {0.0};
  int id_[100] = {0};
  float Egen_[100] = {0.0};

  TH1F *Layer1_energy = new TH1F("Layer1_energy", "", 100, 0.0, 50000.0);
  Layer1_energy->GetXaxis()->SetTitle("Energy in layer 1");  
  TTree *genTree = (TTree*)(infile->Get("Events/Gen"));
  genTree->SetBranchAddress("n_particle", &n_gen);
  genTree->SetBranchAddress("eta", &eta_);
  genTree->SetBranchAddress("phi", &phi_);
  genTree->SetBranchAddress("pT", &pT_);
  genTree->SetBranchAddress("pz", &pz_);
  genTree->SetBranchAddress("pdg_id", &id_);
  genTree->SetBranchAddress("E", &Egen_);

  
  //for(int i=0; i<n_layers; i++){
  //  intree[i] = (TTree*)(infile->Get(Form("Events/layer_%02d", i+1)));
  //  intree[i]->SetBranchAddress("nHit", &nHit_[i]);
  //  intree[i]->SetBranchAddress("X", &X_[i]);
  //  intree[i]->SetBranchAddress("Y", &Y_[i]);
  //  intree[i]->SetBranchAddress("SimHitE", &E_[i]);
  //  intree[i]->SetBranchAddress("time", &t_[i]);
  //  intree[i]->SetBranchAddress("ADC", &adc_[i]);
  //  intree[i]->SetBranchAddress("Thick", &thick_[i]);
  //  intree[i]->SetBranchAddress("ADC_mode", &adc_mode_[i]);
  //  intree[i]->SetBranchAddress("z_side", &zside_[i]);
  //}
  //int nEvents = intree[i]->GetEntriesFast();
  
  // Add scripts for defining histograms
  
  
  for(int j=0; j<n; j++){
    genTree->GetEntry(j);
    float eta_a = 0.0;
    float phi_a = 0.0;
    float pT_a = 0.0;
    float E_a = 0.0;
    float E_HE = 0.0;
    //std::cout<< id_[k] << std::endl;
    for(int k=0; k<n_gen; k++){
      //std::cout<< id_[k] << std::endl;
      if((pz_[k] > 0.0)){
        eta_a = eta_[k];
        phi_a = phi_[k];
        pT_a = pT_[k];
        E_a = Egen_[k];
      }
    }
    if(eta_a < 1.7 || eta_a > 2.7){continue;}
    std::ofstream out_file("CSV_PU_000_Pt_200_Eta_2pt20/Event_fet_"  + std::to_string(j) + ".csv");
    out_file << "X,Y,Layer,Eff_ADC" << std::endl;
    //std::cout<<" Energy " << GenEvent[0].Energy << " GeV" << std::endl;
    int n_node_total = 0;
    float layer_1_ene = 0.0;
    for(int i=0; i<n_layers; i++){
        intree = (TTree*)(infile->Get(Form("Events/layer_%02d", i+1)));
        intree->SetBranchAddress("nHit", &nHit_);
        intree->SetBranchAddress("X", &X_);
        intree->SetBranchAddress("Y", &Y_);
        intree->SetBranchAddress("SimHitE", &E_);
        intree->SetBranchAddress("time", &t_);
        intree->SetBranchAddress("ADC", &adc_);
        intree->SetBranchAddress("Thick", &thick_);
        intree->SetBranchAddress("ADC_mode", &adc_mode_);
        intree->SetBranchAddress("z_side", &zside_);
      intree->GetEntry(j);
      int n_ = 0;
      TLorentzVector p4a;
      p4a.SetPtEtaPhiE(pT_a, eta_a, phi_a, E_a);
      for(int k=0; k<nHit_; k++){
        float ADC_eff = ADC_effective(adc_[k], adc_mode_[k], thick_[k]);
        TLorentzVector p4;
        p4.SetXYZT(X_[k], Y_[k], Z_[i], 0);
        if(zside_[k]>0 && ADC_eff > 13.5 && std::abs(p4.Eta()-eta_a) < 0.25 && std::abs(p4.DeltaPhi(p4a)) < 0.25 && i < 26){
            out_file << p4.Eta()-eta_a << "," << p4.DeltaPhi(p4a) << "," << i+1 << "," << (ADC_eff*p4.CosTheta()) << std::endl;
            n_node_total += 1;
        }
        if(zside_[k]>0 && ADC_eff > 13.5 && std::abs(p4.Eta()-eta_a) < 0.1 && std::abs(p4.DeltaPhi(p4a)) < 0.1 && i >= 26){E_HE += ADC_eff;}
        if(i == 0){
          layer_1_ene += E_[k];
        }
      }
    }
    Layer1_energy->Fill(layer_1_ene);
    out_file.close();
    std::ofstream out_file2("CSV_PU_000_Pt_200_Eta_2pt20/Event_all_"  + std::to_string(j) + ".csv");
    out_file2 << eta_a << "," << phi_a << "," << E_HE << "," << E_a << std::endl;
    out_file2.close(); 
    std::cout<<" Event " << j << " Total Nodes " << n_node_total << std::endl;
  }
  std::unique_ptr<TFile> myFile( TFile::Open("file_60k.root", "RECREATE") );
  myFile->WriteObject(Layer1_energy, "Layer1_energy");
  return 0;
}

int main(){
  CSV_Generation();
}
  
