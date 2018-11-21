#include "AliAnalysisTaskStrangenessLifetimes.h"

#include <array>
#include <unordered_map>

#include <Riostream.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TList.h>
#include <TMath.h>
#include <stdio.h>
#include "AliAnalysisManager.h"
#include "AliESDEvent.h"
#include "AliESDtrack.h"
#include "AliExternalTrackParam.h"
#include "AliInputEventHandler.h"
#include "AliMCEvent.h"
#include "AliPID.h"
#include "AliPDG.h"
#include "AliPIDResponse.h"
#include "AliVVertex.h"

using Lifetimes::MCparticle;
using Lifetimes::MiniV0;
using std::cout;
using std::endl;


ClassImp(AliAnalysisTaskStrangenessLifetimes);

namespace {
constexpr double Sq(double x) { return x * x; }
constexpr float kEps = 1.e-6;

double Distance(double dx, double dy, double dz) {
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

int ComputeMother(AliMCEvent* mcEvent, const AliESDtrack* one, const AliESDtrack* two) {
  int labOne = std::abs(one->GetLabel());
  int labTwo = std::abs(two->GetLabel());

  if (mcEvent->IsPhysicalPrimary(labOne) ||
      mcEvent->IsPhysicalPrimary(labTwo) ||
      mcEvent->IsSecondaryFromMaterial(labOne) ||
      mcEvent->IsSecondaryFromMaterial(labTwo))
    return -1;
  else {
    TParticle* partOne = mcEvent->Particle(labOne);
    TParticle* partTwo = mcEvent->Particle(labTwo);
    if (partOne->GetFirstMother() != partTwo->GetFirstMother()) {
      return -1;
    } else {
      if (one->GetLabel() * two->GetLabel() >= 0)
        return partTwo->GetFirstMother();
      else
        return -partTwo->GetFirstMother();
    }
  }

}

}  // namespace

AliAnalysisTaskStrangenessLifetimes::AliAnalysisTaskStrangenessLifetimes(
    bool mc, std::string name)
    : AliAnalysisTaskSE(name.data()),
      fEventCuts{},
      fListHist{nullptr},
      fTreeV0{nullptr},
      fPIDResponse{nullptr},
      fDoV0Refit{true},
      fMC{mc},
      fUseOnTheFly{false},
      fHistMCct{nullptr},
      fHistMCctPrimary{nullptr},
      fHistMCctSecondaryFromMaterial{nullptr},
      fHistV0radius{nullptr},
      fHistV0pt{nullptr},
      fHistV0eta{nullptr},
      fHistInvMassK0s{nullptr},
      fHistInvMassLambda{nullptr},
      fHistDistOverTotMom{nullptr},
      fHistV0CosPA{nullptr},
      fHistChi2V0{nullptr},
      fHistDcaNeg2PrimaryVertex{nullptr},
      fHistDcaPos2PrimaryVertex{nullptr},
      fHistDcaV0daughters{nullptr},
      fHistV0armAlpha{nullptr},
      fHistV0armPt{nullptr},
      fHistLeastNxedRows{nullptr},
      fHistLeastXedOverFindable{nullptr},
      fHistMaxChi2PerCluster{nullptr},
      fHistNsigmaPosPion{nullptr},
      fHistNsigmaPosProton{nullptr},
      fHistNsigmaNegPion{nullptr},
      fHistNsigmaNegProton{nullptr},
      fHistEtaPos{nullptr},
      fHistEtaNeg{nullptr},
      fHistArmenteros{nullptr},
      fHistNsigmaPosHe{nullptr},
      fHistdEdxVsPt{nullptr},
      fHistNhyp{nullptr},
      fMinPtToSave{0.1},
      fMaxPtToSave{100},
      fMaxTPCpionSigma{10.},
      fMaxTPCprotonSigma{10.},
      fMaxTPChe3Sigma{10.},
      fV0vector{},
      fMCvector{},
      fMultiplicity{} {
  // Standard output
  DefineInput(0, TChain::Class());
  DefineOutput(1, TList::Class());  // Basic Histograms
  DefineOutput(2, TTree::Class());  // V0 Tree output
}

AliAnalysisTaskStrangenessLifetimes::~AliAnalysisTaskStrangenessLifetimes() {
  if (fListHist) {
    delete fListHist;
    fListHist = 0x0;
  }

  if (fTreeV0) {
    delete fTreeV0;
    fTreeV0 = 0x0;
  }
}

void AliAnalysisTaskStrangenessLifetimes::UserCreateOutputObjects() {
  AliAnalysisManager *man = AliAnalysisManager::GetAnalysisManager();
  AliInputEventHandler *inputHandler = (AliInputEventHandler *)(man->GetInputEventHandler());
  fPIDResponse = inputHandler->GetPIDResponse();
  inputHandler->SetNeedField();

  fTreeV0 = new TTree("fTreeV0", "V0 Candidates");
  fTreeV0->Branch("fMultiplicity", &fMultiplicity, "fMultiplicity/F");
  fTreeV0->Branch("V0s", &fV0vector);
  if (man->GetMCtruthEventHandler()) {
    fTreeV0->Branch("MCparticles",&fMCvector);
  }

  fListHist = new TList();
  fListHist->SetOwner();
  fEventCuts.AddQAplotsToList(fListHist);

  fHistV0radius = new TH1D("fHistV0radius", ";V0 r (cm); Counts", 250, 0, 250);
  fHistV0pt =
      new TH1D("fHistV0pt", ";V0 #it{p}_{T} (GeV/#it{c}); Counts", 40, 0., 4.);
  fHistV0eta = new TH1D("fHistV0eta", ";V0 #eta; Counts", 80, -0.8, 0.8);
  fHistInvMassK0s =
      new TH2D("fHistInvMassK0s",
               ";V0 #it{p}_{T} (GeV/#it{c}); m_{#pi#pi} (GeV/#it{c}^{2})", 20,
               0, 2, 80, 0.46, 0.54);
  fHistInvMassLambda =
      new TH2D("fHistInvMassLambda",
               ";V0 #it{p}_{T} (GeV/#it{c}); m_{p#pi} (GeV/#it{c}^{2})", 20, 0,
               2, 80, 1.075, 1.155);
  fHistDistOverTotMom =
      new TH1D("fHistDistOverTotMom", ";V0 L/#it{p} (#it{c} cm / GeV); Counts",
               250, 0, 250);
  fHistV0CosPA = new TH1D("fHistV0CosPA", ";V0 cos(#theta_{P}); Counts",65536,0.9, 1.);
  fHistChi2V0 =
      new TH1D("fHistChi2V0", ";V0 #chi^{2}; Counts",256, 0.,10.);
  fHistDcaNeg2PrimaryVertex =
      new TH1D("fHistDcaNeg2PrimaryVertex", ";Neg prong DCA (cm); Counts",8,0.,0.25);
  fHistDcaPos2PrimaryVertex =
      new TH1D("fHistDcaPos2PrimaryVertex", ";Pos prong DCA (cm); Counts",8,0.,0.25);
  fHistDcaV0daughters = new TH1D(
      "fHistDcaV0daughters", ";Prongs DCA; Counts",8,0.,2.);
  fHistV0armAlpha = new TH1D(
      "fHistV0armAlpha", ";Armenteros #alpha; Counts",8,-1.,1.);
  fHistV0armPt = new TH1D(
      "fHistV0armPt", ";Armenteros #it{p}_{T} (GeV/#it{c}); Counts",8, 0.,0.254);
  fHistLeastNxedRows = new TH1D(
      "fHistLeastNxedRows", ";Min # of crossed rows; Counts", 256, -0.5, 255.5);
  fHistLeastXedOverFindable = new TH1D(
      "fHistLeastXedOverFindable",";Min # of crossed rows / findable clusters; Counts",256, 0. ,1.);
  fHistMaxChi2PerCluster =
      new TH1D("fHistMaxChi2PerCluster", ";Min #chi^{2}/TPC clusters; Counts",256,0., 10.);
  fHistNsigmaPosPion =
      new TH1D("fHistNsigmaPosPion", ";n_{#sigma} TPC Pos Pion; Counts",16, 0.,10.);
  fHistNsigmaPosProton =
      new TH1D("fHistNsigmaPosProton", ";n_{#sigma} TPC Pos Proton; Counts",16,0.,10.);
  fHistNsigmaNegPion =
      new TH1D("fHistNsigmaNegPion", ";n_{#sigma} TPC Neg Pion; Counts",16, 0.,10.);
  fHistNsigmaNegProton =
      new TH1D("fHistNsigmaNegProton", ";n_{#sigma} TPC Neg Proton; Counts",16, 0.,10.);
  fHistEtaPos =
      new TH1D("fHistEtaPos", ";Pos prong #eta; Counts", 128, -1.,1.);
  fHistEtaNeg =
      new TH1D("fHistEtaNeg", ";Neg prong #eta; Counts", 128, -1.,1.);
  fHistArmenteros = new TH2D(
      "fHistArmenteros", ";#alpha;#it{q}_{T}",256,-1., 1., 256, 0.,0.254);

  fHistNsigmaPosHe =
      new TH1D("fHistNsigmaPosHe", ";n_{#sigma} TPC Pos He; Counts",60,0,20);       
  fHistdEdxVsPt = 
      new TH2D("fHistdedxpt",";pt;dedx;counts",100,0,10,100,0,1500);
  fHistNhyp = 
      new TH1D("num of hyper",";pt;counts",60,0,10);

  if (man->GetMCtruthEventHandler()) {
    fHistMCct[0] = new TH1D("fHistMCctK0s", ";MC ct (cm); Counts", 4000, 0, 40);
    fHistMCct[1] = new TH1D("fHistMCctLambda", ";MC ct (cm); Counts", 4000, 0, 40);
    fListHist->Add(fHistMCct[0]);
    fListHist->Add(fHistMCct[1]);

    fHistMCctPrimary[0] = new TH1D("fHistMCctPrimaryK0s", ";MC ct (cm); Counts", 4000, 0, 40);
    fHistMCctPrimary[1] = new TH1D("fHistMCctPrimaryLambda", ";MC ct (cm); Counts", 4000, 0, 40);
    fListHist->Add(fHistMCctPrimary[0]);
    fListHist->Add(fHistMCctPrimary[1]);

    fHistMCctSecondaryFromMaterial[0] = new TH1D("fHistMCctSecondaryFromMaterialK0s", ";MC ct (cm); Counts", 4000, 0, 40);
    fHistMCctSecondaryFromMaterial[1] = new TH1D("fHistMCctSecondaryFromMaterialLambda", ";MC ct (cm); Counts", 4000, 0, 40);
    fListHist->Add(fHistMCctSecondaryFromMaterial[0]);
    fListHist->Add(fHistMCctSecondaryFromMaterial[1]);
  }
  fListHist->Add(fHistNhyp);
  fListHist->Add(fHistdEdxVsPt);
  fListHist->Add(fHistNsigmaPosHe);
  fListHist->Add(fHistV0radius);
  fListHist->Add(fHistV0pt);
  fListHist->Add(fHistV0eta);
  fListHist->Add(fHistInvMassK0s);
  fListHist->Add(fHistInvMassLambda);
  fListHist->Add(fHistDistOverTotMom);
  fListHist->Add(fHistV0CosPA);
  fListHist->Add(fHistChi2V0);
  fListHist->Add(fHistDcaNeg2PrimaryVertex);
  fListHist->Add(fHistDcaPos2PrimaryVertex);
  fListHist->Add(fHistDcaV0daughters);
  fListHist->Add(fHistV0armAlpha);
  fListHist->Add(fHistV0armPt);
  fListHist->Add(fHistLeastNxedRows);
  fListHist->Add(fHistLeastXedOverFindable);
  fListHist->Add(fHistMaxChi2PerCluster);
  fListHist->Add(fHistNsigmaPosPion);
  fListHist->Add(fHistNsigmaPosProton);
  fListHist->Add(fHistNsigmaNegPion);
  fListHist->Add(fHistNsigmaNegProton);
  fListHist->Add(fHistEtaPos);
  fListHist->Add(fHistEtaNeg);
  fListHist->Add(fHistArmenteros);

  PostData(1, fListHist);
  PostData(2, fTreeV0);

  AliPDG::AddParticlesToPdgDataBase();
}  // end UserCreateOutputObjects

void AliAnalysisTaskStrangenessLifetimes::UserExec(Option_t *) {
  AliESDEvent *esdEvent = dynamic_cast<AliESDEvent *>(InputEvent());
  if (!esdEvent) {
    ::Fatal("AliAnalysisTaskStrangenessLifetimes::UserExec",
            "AliESDEvent not found.");
    return;
  }

  std::array<int,3> pdgCodes{310, 3122, 1010010030};
  AliMCEvent* mcEvent = MCEvent();
  if (!mcEvent && fMC) {
    ::Fatal("AliAnalysisTaskStrangenessLifetimes::UserExec","Could not retrieve MC event");
    return;
  }

  double magneticField = esdEvent->GetMagneticField();

  if (!fEventCuts.AcceptEvent(esdEvent)) {
    PostData(1, fListHist);
    PostData(2, fTreeV0);
    return;
  }

  double primaryVertex[3];
  fMultiplicity = fEventCuts.GetCentrality();
  fEventCuts.GetPrimaryVertex()->GetXYZ(primaryVertex);

  std::unordered_map<int,int> mcMap;
  if (fMC) {
    const AliVVertex* mcV = mcEvent->GetPrimaryVertex();
    fMCvector.clear();
    for (int ilab = 0;  ilab < mcEvent->GetNumberOfTracks(); ilab++) {   // This is the begining of the loop on tracks
      TParticle* part = mcEvent->Particle( ilab );
      if(!part) {
        ::Warning("AliAnalysisTaskStrangenessLifetimes::UserExec","Generated loop %d - MC TParticle pointer to current stack particle = 0x0 ! Skipping.", ilab );
        continue;
      }
      
      int currentPDG = part->GetPdgCode();
      int idx = 0;
      for (auto code : pdgCodes) {
        if (code == std::abs(currentPDG)) {
          if (std::abs(part->Y()) > 1.) {
            continue;
          }

          double sVtx[3]={0.};
          if (part->GetFirstDaughter() == part->GetLastDaughter())
            continue;
          for (int iD = part->GetFirstDaughter(); iD <= part->GetLastDaughter(); ++iD) {
            TParticle* dau = mcEvent->Particle(iD);
            if (mcEvent->IsSecondaryFromWeakDecay(iD) && dau) {
              sVtx[0] = dau->Vx();
              sVtx[1] = dau->Vy();
              sVtx[2] = dau->Vz();
              break;
            }
          }
          double dist = Distance(mcV->GetX() - sVtx[0], mcV->GetY() - sVtx[1], mcV->GetZ() - sVtx[2]);
          double radius = std::hypot(sVtx[0], sVtx[1]);

          MCparticle v0part;
          v0part.SetPDGcode(currentPDG);
          v0part.SetEta(part->Eta());
          v0part.SetPt(part->Pt());
          v0part.SetDistOverP(dist / part->P() + 1e-16);
          v0part.SetRadius(radius);
          bool isSecondary = mcEvent->IsSecondaryFromWeakDecay(ilab);
          fHistMCct[idx]->Fill(dist * part->GetMass() / part->P());
          TParticle* mother = mcEvent->Particle(part->GetFirstMother());
          if (isSecondary && mother) {
            v0part.SetStatus(MCparticle::kSecondaryFromWeakDecay);

            double motherDist = Distance(mcV->GetX() - part->Vx(), mcV->GetY() - part->Vy(), mcV->GetZ() - part->Vz());
            double motherR = std::hypot(part->Vx(), part->Vy());
            MCparticle motherPart;
            motherPart.SetPDGcode(mother->GetPdgCode());
            motherPart.SetEta(mother->Eta());
            motherPart.SetPt(mother->Pt());
            motherPart.SetDistOverP(motherDist / mother->P() + 1e-16);
            motherPart.SetRadius(motherR);
            fMCvector.push_back(motherPart);
          } else if (mcEvent->IsPhysicalPrimary(ilab)) {
            v0part.SetStatus(MCparticle::kPrimary);
            fHistMCctPrimary[idx]->Fill(dist * part->GetMass() / part->P());
          } else if (mcEvent->IsSecondaryFromMaterial(ilab)) {
            v0part.SetStatus(MCparticle::kSecondaryFromMaterial);
            fHistMCctSecondaryFromMaterial[idx]->Fill(dist * part->GetMass() / part->P());
          } else {
            ::Fatal("AliAnalysisTaskStrangenessLifetimes::UserExec",
              "A particle that is not primary, not secondary from weak decay nor from material."
              "It does know only what it is not.");
          }
          mcMap[ilab] = fMCvector.size();
          if (v0part.GetPDGcode()==pdgCodes[2]) {
             if( (part->GetLastDaughter()-part->GetFirstDaughter())==2 ) v0part.SetNBodies(3);
             else{ v0part.SetNBodies(2);
             fHistNhyp->Fill(v0part.GetPt());}
          }
          fMCvector.push_back(v0part);
        }
        ++idx;
      }
    }
  }

  fV0vector.clear();
  for (int iV0 = 0; iV0 < esdEvent->GetNumberOfV0s();
       iV0++) {  // This is the begining of the V0 loop (we analyse only offline
                 // V0s)
    AliESDv0 *v0 = ((AliESDEvent *)esdEvent)->GetV0(iV0);
    if (!v0) continue;
    if (v0->GetOnFlyStatus() != 0 && !fUseOnTheFly) continue;
    if (fUseOnTheFly && v0->GetOnFlyStatus() == 0) continue;

    // Remove like-sign (will not affect offline V0 candidates!)
    if (v0->GetParamN()->Charge() * v0->GetParamP()->Charge() > 0) continue;

    const int lKeyPos = std::abs(v0->GetPindex());
    const int lKeyNeg = std::abs(v0->GetNindex());
    AliESDtrack *pTrack = esdEvent->GetTrack(lKeyPos);
    AliESDtrack *nTrack = esdEvent->GetTrack(lKeyNeg);

    if (fMC) {
      AliESDtrack* one = esdEvent->GetTrack(v0->GetNindex());
      AliESDtrack* two = esdEvent->GetTrack(v0->GetPindex());
      if (!one || !two)
        ::Fatal("AliAnalysisTaskStrangenessLifetimes::UserExec",
          "Missing V0 tracks %p %p",(void*)one,(void*)two);
      int ilab = std::abs(ComputeMother(mcEvent, one, two));
      TParticle* part = mcEvent->Particle(ilab);
        
      int currentPDG = part->GetPdgCode();
        
      if (currentPDG==pdgCodes[2] && (part->GetLastDaughter()-part->GetFirstDaughter())==1) {
          fHistNsigmaPosHe->Fill(std::abs(fPIDResponse->NumberOfSigmasTPC(pTrack, AliPID::kHe3)));
          fHistdEdxVsPt->Fill(pTrack->GetTPCmomentum(),pTrack->GetTPCsignal());

      }
        
    }    
    

    // Official means of acquiring N-sigmas
    float nSigmaPosProton =
        std::abs(fPIDResponse->NumberOfSigmasTPC(pTrack, AliPID::kProton));
    float nSigmaPosPion =
        std::abs(fPIDResponse->NumberOfSigmasTPC(pTrack, AliPID::kPion));
    float nSigmaPosHe3 =
        std::abs(fPIDResponse->NumberOfSigmasTPC(pTrack, AliPID::kHe3));
    float nSigmaNegProton =
        std::abs(fPIDResponse->NumberOfSigmasTPC(nTrack, AliPID::kProton));
    float nSigmaNegPion =
        std::abs(fPIDResponse->NumberOfSigmasTPC(nTrack, AliPID::kPion));
    float nSigmaNegHe3 =
        std::abs(fPIDResponse->NumberOfSigmasTPC(nTrack, AliPID::kHe3));

    bool isHyperCandidate = nSigmaNegHe3 < 5 || nSigmaPosHe3 < 5;
    double v0Pt = v0->Pt();

    if ((v0Pt < fMinPtToSave) || (fMaxPtToSave < v0Pt)) continue;

    double decayVtx[3];
    v0->GetXYZ(decayVtx[0], decayVtx[1], decayVtx[2]);

    double tV0mom[3];
    v0->GetPxPyPz(tV0mom[0], tV0mom[1], tV0mom[2]);
    double lV0TotalMomentum = std::sqrt(
        tV0mom[0] * tV0mom[0] + tV0mom[1] * tV0mom[1] + tV0mom[2] * tV0mom[2]);

    double v0Radius = std::hypot(decayVtx[0], decayVtx[1]);

    double momPos[3];
    v0->GetPPxPyPz(momPos[0], momPos[1], momPos[2]);
    double momNeg[3];
    v0->GetNPxPyPz(momNeg[0], momNeg[1], momNeg[2]);

    // Calculate the sign of the vec prod with momenta projected to xy plane
    // It is unnecessary to to the full calculation like done in the original
    // task
    double lVecProd = momPos[0] * momNeg[1] - momPos[1] * momNeg[0];
    bool isCowboy = lVecProd * magneticField < 0;

   // AliESDtrack *pTrack = esdEvent->GetTrack(lKeyPos);
   // AliESDtrack *nTrack = esdEvent->GetTrack(lKeyNeg);

    if (!pTrack || !nTrack) {
      ::Fatal("AliAnalysisTaskStrangenessLifetimes::UserExec",
              "Could not retreive one of the daughter track");
      continue;
    }

    /// TODO: check if this extra cleanup is required
    if (std::abs(nTrack->Eta()) > 0.8 || std::abs(pTrack->Eta()) > 0.8)
      continue;
    if (std::abs(v0->RapK0Short()) > 0.5 && std::abs(v0->RapLambda()) > 0.5)
      continue;

    // Filter like-sign V0 (next: add counter and distribution)
    if (pTrack->GetSign() == nTrack->GetSign()) {
      continue;
    }

    // Track quality cuts
    unsigned char posXedRows = pTrack->GetTPCClusterInfo(2, 1);
    unsigned char negXedRows = nTrack->GetTPCClusterInfo(2, 1);

    // TPC refit condition (done during reconstruction for Offline but not for
    // On-the-fly)
    if (!(pTrack->GetStatus() & AliESDtrack::kTPCrefit)) continue;
    if (!(nTrack->GetStatus() & AliESDtrack::kTPCrefit)) continue;

    float negB[2], posB[2], bCov[3];
    pTrack->GetImpactParameters(posB, bCov);
    nTrack->GetImpactParameters(negB, bCov);

    // GetKinkIndex condition
    if (pTrack->GetKinkIndex(0) > 0 || nTrack->GetKinkIndex(0) > 0) continue;

    // Findable cluster s > 0 condition
    if (pTrack->GetTPCNclsF() <= 0 || nTrack->GetTPCNclsF() <= 0) continue;

    float posXedRowsOverFindable = float(posXedRows) / pTrack->GetTPCNclsF();
    float negXedRowsOverFindable = float(negXedRows) / nTrack->GetTPCNclsF();

    float posChi2PerCluster =
        pTrack->GetTPCchi2() / (pTrack->GetTPCNcls() + 1.e-16);
    float negChi2PerCluster =
        nTrack->GetTPCchi2() / (nTrack->GetTPCNcls() + 1.e-16);

    // Extra track quality: min track length
    float posTrackLength = -1;
    float negTrackLength = -1;
    if (pTrack->GetInnerParam())
      posTrackLength = pTrack->GetLengthInActiveZone(
          1, 2.0, 220.0, esdEvent->GetMagneticField());
    if (nTrack->GetInnerParam())
      negTrackLength = nTrack->GetLengthInActiveZone(
          1, 2.0, 220.0, esdEvent->GetMagneticField());

    float smallestTrackLength =
        (posTrackLength < negTrackLength) ? posTrackLength : negTrackLength;
    if ((((pTrack->GetTPCClusterInfo(2, 1)) < 70) ||
         ((nTrack->GetTPCClusterInfo(2, 1)) < 70)) &&
        smallestTrackLength < 80)
      continue;

    double dcaPosToPrimVertex = std::abs(
        pTrack->GetD(primaryVertex[0], primaryVertex[1], magneticField));

    double dcaNegToPrimVertex = std::abs(
        nTrack->GetD(primaryVertex[0], primaryVertex[1], magneticField));

    double cosPA = v0->GetV0CosineOfPointingAngle(
        primaryVertex[0], primaryVertex[1], primaryVertex[2]);
    if (cosPA < 0.9) continue;

    // Getting invariant mass infos directly from ESD
    double masses[3];
    for (int iPdg = 0; iPdg < 3; ++iPdg) {
      auto lvector = GetV0LorentzVector(pdgCodes[iPdg], nTrack, pTrack, v0->AlphaV0());
      masses[iPdg] = lvector.M();
      if (iPdg == 2) {
        if (isHyperCandidate) {
          v0Pt = lvector.Pt();
          lV0TotalMomentum = lvector.P();
        } else
          masses[iPdg] = -1;
      }
    }

 

    float distOverP = std::sqrt(Sq(decayVtx[0] - primaryVertex[0]) +
                                Sq(decayVtx[1] - primaryVertex[1]) +
                                Sq(decayVtx[2] - primaryVertex[2])) /
                      (lV0TotalMomentum + 1e-16);  // avoid division by zero
    unsigned char minXedRows =
        posXedRows < negXedRows ? posXedRows : negXedRows;
    float minXedRowsOverFindable =
        posXedRowsOverFindable < negXedRowsOverFindable
            ? posXedRowsOverFindable
            : negXedRowsOverFindable;
    float maxChi2PerCluster = posChi2PerCluster > negChi2PerCluster
                                  ? posChi2PerCluster
                                  : negChi2PerCluster;

    bool negTOF = nTrack->GetTOFsignal() * 1.e-3 < 100; // in ns, loose cut on TOF beta (<~0.2)
    bool posTOF = pTrack->GetTOFsignal() * 1.e-3 < 100; // in ns, loose cut on TOF beta (<~0.2)

    bool posITSrefit = pTrack->GetStatus() & AliESDtrack::kITSrefit;
    bool negITSrefit = nTrack->GetStatus() & AliESDtrack::kITSrefit;
    bool posSPDany = pTrack->HasPointOnITSLayer(0) || pTrack->HasPointOnITSLayer(1);
    bool negSPDany = nTrack->HasPointOnITSLayer(0) || nTrack->HasPointOnITSLayer(1);
    int ITSnCl = (nTrack->GetITSclusters(0) > pTrack->GetITSclusters(0)) ? pTrack->GetITSclusters(0) : nTrack->GetITSclusters(0);


    // Rugh 20-sigma selection band, parametric.
    // K0Short: Enough to parametrize peak broadening with linear function.
    double lUpperLimitK0Short = (5.63707e-01) + (1.14979e-02) * v0Pt;
    double lLowerLimitK0Short = (4.30006e-01) - (1.10029e-02) * v0Pt;
    // Lambda: Linear (for higher pt) plus exponential (for low-pt broadening)
    //[0]+[1]*x+[2]*TMath::Exp(-[3]*x)
    double upperLimitLambda = (1.13688e+00) + (5.27838e-03) * v0Pt +
                              (8.42220e-02) * TMath::Exp(-(3.80595e+00) * v0Pt);
    double lowerLimitLambda = (1.09501e+00) - (5.23272e-03) * v0Pt -
                              (7.52690e-02) * TMath::Exp(-(3.46339e+00) * v0Pt);
    // Do Selection
    if (
        // Case 1: Lambda Selection
        (masses[1] < upperLimitLambda && masses[1] > lowerLimitLambda &&
         ((nSigmaPosProton < fMaxTPCprotonSigma &&
           nSigmaNegPion < fMaxTPCpionSigma) ||
          (nSigmaNegProton < fMaxTPCprotonSigma &&
           nSigmaPosPion < fMaxTPCpionSigma))) ||
        // Case 2: K0Short Selection
        (masses[0] < lUpperLimitK0Short && masses[0] > lLowerLimitK0Short &&
         nSigmaNegPion < fMaxTPCpionSigma &&
         nSigmaPosPion < fMaxTPCpionSigma) ||
        // Case 3: Hypertriton Selection        
        (masses[2] > 2.85 && masses[2] < 3.15 &&
          ((nSigmaPosHe3 < fMaxTPChe3Sigma &&
           nSigmaNegPion < fMaxTPCpionSigma) ||
          (nSigmaNegHe3 < fMaxTPChe3Sigma &&
           nSigmaPosPion < fMaxTPCpionSigma)))) {
      /// Filling monitoring histograms
      fHistV0radius->Fill(v0Radius);
      fHistV0pt->Fill(v0Pt);
      fHistV0eta->Fill(v0->Eta());
      fHistInvMassK0s->Fill(v0Pt, masses[0]);
      fHistInvMassLambda->Fill(v0Pt, masses[1]);
      fHistDistOverTotMom->Fill(distOverP);
      fHistV0CosPA->Fill(cosPA);
      fHistChi2V0->Fill(v0->GetChi2V0());
      fHistDcaNeg2PrimaryVertex->Fill(dcaNegToPrimVertex);
      fHistDcaPos2PrimaryVertex->Fill(dcaPosToPrimVertex);
      fHistDcaV0daughters->Fill(v0->GetDcaV0Daughters());
      fHistV0armAlpha->Fill(v0->AlphaV0());
      fHistV0armPt->Fill(v0->PtArmV0());
      fHistLeastNxedRows->Fill(minXedRows);
      fHistLeastXedOverFindable->Fill(minXedRowsOverFindable);
      fHistMaxChi2PerCluster->Fill(maxChi2PerCluster);
      fHistNsigmaPosPion->Fill(nSigmaPosPion);
      fHistNsigmaPosProton->Fill(nSigmaPosProton);
      fHistNsigmaNegPion->Fill(nSigmaNegPion);
      fHistNsigmaNegProton->Fill(nSigmaNegProton);
      fHistEtaPos->Fill(pTrack->Eta());
      fHistEtaNeg->Fill(nTrack->Eta());
      fHistArmenteros->Fill(v0->AlphaV0(), v0->PtArmV0());

      // Filling the V0 vector
      MiniV0 miniV0;
      miniV0.SetV0ptAndFake(v0Pt, false);
      miniV0.SetV0eta(v0->Eta());
      miniV0.SetLeastNumberOfXedRows(minXedRows);
      miniV0.SetDistOverP(distOverP);
      for (int iPdg = 0; iPdg < 3; ++iPdg)
        miniV0.SetInvMass(iPdg, masses[iPdg]);
      miniV0.SetArmenterosVariables(v0->AlphaV0(), v0->PtArmV0());
      miniV0.SetV0CosPA(cosPA);
      miniV0.SetV0Chi2andCowBoy(v0->GetChi2V0(), isCowboy);
      miniV0.SetProngsDCA(v0->GetDcaV0Daughters());
      miniV0.SetProngsPvDCA(dcaPosToPrimVertex, dcaNegToPrimVertex);
      miniV0.SetV0radiusAndLikeSign(v0Radius);
      miniV0.SetLeastXedRowsOverFindable(minXedRowsOverFindable);
      miniV0.SetMaxChi2perCluster(maxChi2PerCluster);
      miniV0.SetProngsEtaTOF(pTrack->Eta(), posTOF, nTrack->Eta(), negTOF);
      miniV0.SetProngsTPCnsigmas(nSigmaPosPion, nSigmaPosProton,
                                 nSigmaNegPion, nSigmaNegProton);
      miniV0.SetITSinformation(negITSrefit, posITSrefit, negSPDany, posSPDany, ITSnCl);

      if (fMC) {
        AliESDtrack* one = esdEvent->GetTrack(v0->GetNindex());
        AliESDtrack* two = esdEvent->GetTrack(v0->GetPindex());
        if (!one || !two)
          ::Fatal("AliAnalysisTaskStrangenessLifetimes::UserExec",
            "Missing V0 tracks %p %p",(void*)one,(void*)two);
        int ilab = std::abs(ComputeMother(mcEvent, one, two));
        TParticle* part = mcEvent->Particle(ilab);
        if(!part) {
          continue;
        }
        int currentPDG = part->GetPdgCode();
        for (auto code : pdgCodes) {
          if (code == std::abs(currentPDG)) {
            fMCvector[mcMap[ilab]].SetRecoIndex(fV0vector.size());
            break;
          }
        }
      }
      fV0vector.push_back(miniV0);

    }
  }

  if (fV0vector.size() || fMCvector.size()) fTreeV0->Fill();

  PostData(1, fListHist);
  PostData(2, fTreeV0);
}

void AliAnalysisTaskStrangenessLifetimes::Terminate(Option_t *) {}

LVector_t AliAnalysisTaskStrangenessLifetimes::GetV0LorentzVector(int pdg, AliESDtrack* negTrack, AliESDtrack* posTrack, double alpha) {
  constexpr int v0Pdg[3]{310, 3122, 1010010030};
  constexpr AliPID::EParticleType children[3][2]{
    {AliPID::kPion, AliPID::kPion},
    {AliPID::kProton, AliPID::kPion},
    {AliPID::kHe3, AliPID::kPion},
  };
  for (int iPdg = 0; iPdg < 3; ++iPdg) {
    if (pdg != v0Pdg[iPdg]) continue;
    int posIndex = int(alpha < 0);
    int negIndex = int(alpha >= 0);
    double posMass = AliPID::ParticleMass(children[iPdg][posIndex]);
    double negMass = AliPID::ParticleMass(children[iPdg][negIndex]);
    int posCharge = AliPID::ParticleCharge(children[iPdg][posIndex]);
    int negCharge = AliPID::ParticleCharge(children[iPdg][negIndex]);
    double posMom[3],negMom[3];
    posTrack->GetPxPyPz(posMom);
    negTrack->GetPxPyPz(negMom);
    LVector_t posLvec{posMom[0] * posCharge, posMom[1] * posCharge, posMom[2], posMass};
    LVector_t negLvec{negMom[0] * negCharge, negMom[1] * negCharge, negMom[2], negMass};
    posLvec += negLvec;
    return posLvec;
  }
  return LVector_t(0,0,0,0);
} 
