#include "AliAnalysisTaskHypertriton3.h"

#include "AliAnalysisDataContainer.h"
#include "AliAnalysisManager.h"
#include "AliESDEvent.h"
#include "AliESDtrack.h"
#include "AliInputEventHandler.h"
#include "AliMCEvent.h"
#include "AliPDG.h"
#include "AliPID.h"
#include "AliPIDResponse.h"
#include "AliVVertex.h"

#include <Riostream.h>
#include <TChain.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TList.h>
#include <TRandom3.h>

#include <array>
#include <cmath>
#include <vector>
#include <unordered_map>

#include "Math/GenVector/Boost.h"
#include "Math/Vector3Dfwd.h"
#include "Math/Vector3D.h"
#include "Math/LorentzVector.h"

#include "AliDataFile.h"
#include <TFile.h>
#include <TSpline.h>

#include "Track.h"
#include <memory>

#define HomogeneousField

ClassImp(AliAnalysisTaskHypertriton3);

namespace
{

  using lVector = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>;

  struct HelperParticle
  {
    o2::track::TrackParCov *track = nullptr;
    float nSigmaTPC = -1.f;
    float nSigmaTOF = -1.f;
  };

  constexpr float kDeuMass{1.87561};
  constexpr float kPMass{0.938272};
  constexpr float kPiMass{0.13957};
  constexpr float kMasses[3]{kDeuMass, kPMass, kPiMass};
  constexpr AliPID::EParticleType kAliPID[3]{AliPID::kDeuteron, AliPID::kProton, AliPID::kPion};
  const int kPDGs[3]{AliPID::ParticleCode(kAliPID[0]), AliPID::ParticleCode(kAliPID[1]), AliPID::ParticleCode(kAliPID[2])};

  bool IsHyperTriton3(const AliVParticle *vPart, AliMCEvent *mcEvent)
  {
    int nDaughters = 0;

    int vPartPDG = vPart->PdgCode();
    int vPartLabel = vPart->GetLabel();

    if (!mcEvent->IsPhysicalPrimary(vPartLabel) || (std::abs(vPartPDG) != 1010010030))
      return false;

    for (int iD = vPart->GetDaughterFirst(); iD <= vPart->GetDaughterLast(); iD++)
    {
      AliVParticle *dPart = mcEvent->GetTrack(iD);

      int dPartPDG = dPart->PdgCode();
      if (std::abs(dPartPDG) != 11)
        nDaughters++;
    }
    if (nDaughters == 3)
      return true;
    return false;
  }

  int IsTrueHyperTriton3Candidate(AliESDtrack *t1, AliESDtrack *t2, AliESDtrack *t3, AliMCEvent *mcEvent)
  {
    if (!mcEvent)
      return 0;

    int lab1 = std::abs(t1->GetLabel());
    int lab2 = std::abs(t2->GetLabel());
    int lab3 = std::abs(t3->GetLabel());

    if (mcEvent->IsPhysicalPrimary(lab1))
      return -1;
    if (mcEvent->IsPhysicalPrimary(lab2))
      return -1;
    if (mcEvent->IsPhysicalPrimary(lab3))
      return -1;

    AliVParticle *part1 = mcEvent->GetTrack(lab1);
    AliVParticle *part2 = mcEvent->GetTrack(lab2);
    AliVParticle *part3 = mcEvent->GetTrack(lab3);

    if (!part1 || !part2 || !part3)
      return -1;

    int mom1 = part1->GetMother();
    int mom2 = part2->GetMother();
    int mom3 = part3->GetMother();

    if (mom1 != mom2 || mom1 != mom3 || mom2 != mom3)
      return -1;

    AliVParticle *mom = mcEvent->GetTrack(mom1);
    if (!mom)
      return -1;

    return (IsHyperTriton3(mom, mcEvent)) ? mom1 : -1;
  }

  bool HasTOF(AliVTrack *track)
  {
    const bool hasTOFout = track->GetStatus() & AliVTrack::kTOFout;
    const bool hasTOFtime = track->GetStatus() & AliVTrack::kTIME;
    return hasTOFout && hasTOFtime;
  }

} // namespace

AliAnalysisTaskHypertriton3::AliAnalysisTaskHypertriton3(bool mc, std::string name)
    : AliAnalysisTaskSE(name.data()), fEventCuts{}, fVertexer{}, fVertexerLambda{}, fMC{mc}
{
  fTrackCuts.SetMinNClustersTPC(0);
  fTrackCuts.SetEtaRange(-0.9, 0.9);
  /// Settings for the custom vertexer

  /// Standard output
  DefineInput(0, TChain::Class());
  DefineOutput(1, TList::Class()); // Basic Histograms
  DefineOutput(2, TTree::Class()); // Hypertriton Candidates Tree output
}

AliAnalysisTaskHypertriton3::~AliAnalysisTaskHypertriton3()
{
  if (fListHist)
  {
    delete fListHist;
    fListHist = nullptr;
  }

  if (fTreeHyp3)
  {
    delete fTreeHyp3;
    fTreeHyp3 = nullptr;
  }

  if (fCosPAsplineFile)
    delete fCosPAsplineFile;

  if (fGenHypO2)
  {
    delete fGenHypO2;
  }
  else if (o2RecHyp)
    delete o2RecHyp;
}

void AliAnalysisTaskHypertriton3::UserCreateOutputObjects()
{
  AliAnalysisManager *man = AliAnalysisManager::GetAnalysisManager();
  fInputHandler = (AliInputEventHandler *)(man->GetInputEventHandler());
  fPIDResponse = fInputHandler->GetPIDResponse();

  fInputHandler->SetNeedField();

  fListHist = new TList();
  fListHist->SetOwner(true);
  fEventCuts.AddQAplotsToList(fListHist);

  fHistNSigmaDeu = new TH2D("fHistNSigmaDeu", ";#it{p}_{T} (GeV/#it{c});n_{#sigma} TPC Deuteron; Counts", 100, 0., 10.,
                            80, -5.0, 5.0);
  fHistNSigmaP =
      new TH2D("fHistNSigmaP", ";#it{p}_{T} (GeV/#it{c});n_{#sigma} TPC Proton; Counts", 100, 0., 10., 80, -5.0, 5.0);
  fHistNSigmaPi =
      new TH2D("fHistNSigmaPi", ";#it{p}_{T} (GeV/#it{c});n_{#sigma} TPC Pion; Counts", 100, 0., 10., 80, -5.0, 5.0);

  fHistInvMass =
      new TH2D("fHistInvMass", ";m_{dp#pi}(GeV/#it{c^2}); #it{p}_{T} (GeV/#it{c}); Counts", 30, 2.96, 3.05, 100, 0, 10);

  fListHist->Add(fHistNSigmaDeu);
  fListHist->Add(fHistNSigmaP);
  fListHist->Add(fHistNSigmaPi);

  fListHist->Add(fHistInvMass);

  OpenFile(2);
  fTreeHyp3 = new TTree("Hyp3O2", "Hypetriton 3 Body with the O2 Vertexer");

  if (fMC && man->GetMCtruthEventHandler())
  {

    fGenHypO2 = new SHyperTriton3O2;
    o2RecHyp = (RHyperTriton *)fGenHypO2;
    fTreeHyp3->Branch("SHyperTriton", fGenHypO2);
  }
  else
  {

    o2RecHyp = new RHyperTriton3O2;
    fTreeHyp3->Branch("RHyperTriton", static_cast<RHyperTriton3O2 *>(o2RecHyp));
  }
  fCosPAsplineFile = TFile::Open(AliDataFile::GetFileName(fCosPAsplineName).data());
  if (fCosPAsplineFile)
  {
    fCosPAspline = (TSpline3 *)fCosPAsplineFile->Get("cutSpline");
  }

  PostData(1, fListHist);
  PostData(2, fTreeHyp3);

  AliPDG::AddParticlesToPdgDataBase();

} /// end UserCreateOutputObjects

void AliAnalysisTaskHypertriton3::UserExec(Option_t *)
{
  // set Magnetic field for KF
  AliESDEvent *esdEvent = dynamic_cast<AliESDEvent *>(InputEvent());
  if (!esdEvent)
  {
    ::Fatal("AliAnalysisTaskHypertriton3::UserExec", "AliESDEvent not found.");
    return;
  }

  AliMCEvent *mcEvent = MCEvent();
  if (!mcEvent && fMC)
  {
    ::Fatal("AliAnalysisTaskHypertriton3::UserExec", "Could not retrieve MC event");
    return;
  }

  if (!fEventCuts.AcceptEvent(esdEvent))
  {
    PostData(1, fListHist);
    PostData(2, fTreeHyp3);
    return;
  }

  if (!fMC && fDownscaling)
  {
    if (gRandom->Rndm() > fDownscalingFactorByEvent)
      return;
  }

  double pvPos[3], pvCov[6];
  fEventCuts.GetPrimaryVertex()->GetXYZ(pvPos);
  fEventCuts.GetPrimaryVertex()->GetCovarianceMatrix(pvCov);
  o2RecHyp->centrality = fEventCuts.GetCentrality();

  o2RecHyp->trigger = 0u;
  if (fInputHandler->IsEventSelected() & AliVEvent::kINT7)
    o2RecHyp->trigger |= kINT7;
  if (fInputHandler->IsEventSelected() & AliVEvent::kCentral)
    o2RecHyp->trigger |= kCentral;
  if (fInputHandler->IsEventSelected() & AliVEvent::kSemiCentral)
    o2RecHyp->trigger |= kSemiCentral;
  if (fInputHandler->IsEventSelected() & AliVEvent::kHighMultV0)
    o2RecHyp->trigger |= kHighMultV0;
  o2RecHyp->trigger |= esdEvent->GetMagneticField() > 0 ? kPositiveB : 0;

  std::vector<HelperParticle> helpers[3][2];
  std::vector<EventMixingTrack> deuteronsForMixing;
  for (int iTrack = 0; iTrack < esdEvent->GetNumberOfTracks(); iTrack++)
  {
    AliESDtrack *track = esdEvent->GetTrack(iTrack);
    if (!track)
      continue;

    if (!fTrackCuts.AcceptTrack(track))
      continue;

    if (fMC && fOnlyTrueCandidates)
    {
      int lab = std::abs(track->GetLabel());
      if (!mcEvent->IsSecondaryFromWeakDecay(lab))
        continue;
      AliVParticle *part = mcEvent->GetTrack(lab);
      AliVParticle *moth = mcEvent->GetTrack(part->GetMother());
      if (std::abs(moth->PdgCode()) != 1010010030)
        continue;
    }

    bool candidate[3]{false, false, false};
    float nSigmasTPC[3]{-1., -1., -1.}, nSigmasTOF[3]{-1., -1., -1.};
    bool hasTOF{HasTOF(track)};
    float dca[2];
    track->GetImpactParameters(dca[0], dca[1]);
    double dcaNorm = std::hypot(dca[0], dca[1]);

    if (fUseCovarianceCut)
    {
      float cyy = track->GetSigmaY2(), czz = track->GetSigmaZ2(), cyz = track->GetSigmaZY();
      float detYZ = cyy * czz - cyz * cyz;
      if (detYZ < 0.)
        continue;
    }

    for (int iT{0}; iT < 3; ++iT)
    {
      nSigmasTPC[iT] = fPIDResponse->NumberOfSigmasTPC(track, kAliPID[iT]);
      nSigmasTOF[iT] = fPIDResponse->NumberOfSigmasTOF(track, kAliPID[iT]);
      bool requireTOFpid = track->P() > fRequireTOFpid[iT];
      if (std::abs(nSigmasTPC[iT]) < fTPCsigmas[iT] && dcaNorm > fMinTrackDCA[iT] && track->Pt() < fTrackPtRange[iT][1] &&
          track->Pt() > fTrackPtRange[iT][0] && track->GetTPCsignalN() >= fMinTPCpidClusters[iT])
        candidate[iT] = (std::abs(nSigmasTOF[iT]) < fTOFsigmas[iT]) || (!hasTOF && !requireTOFpid);
    }

    if (candidate[0] || candidate[1] || candidate[2])
    {
      HelperParticle helper;
      helper.track = static_cast<o2::track::TrackParCov *>((AliExternalTrackParam *)track);
      for (int iT{0}; iT < 3; ++iT)
      {
        if (candidate[iT])
        {
          int chargeIndex = (fSwapSign && iT == fMixingTrack) ? track->GetSigned1Pt() < 0 : track->GetSigned1Pt() > 0;
          helper.nSigmaTPC = nSigmasTPC[iT];
          helper.nSigmaTOF = nSigmasTOF[iT];

          if (iT == fMixingTrack && fEnableEventMixing)
            deuteronsForMixing.emplace_back(track, nSigmasTPC[iT], nSigmasTOF[iT], 0);
          else
            helpers[iT][chargeIndex].push_back(helper);
        }
      }
    }
  }

  if (fEnableEventMixing)
  {
    auto mixingDeuterons = GetEventMixingTracks(fEventCuts.GetCentrality(), pvPos[2]);
    for (auto mixTrack : mixingDeuterons)
    {
      HelperParticle helper;
      AliESDtrack *track = &(mixTrack->track);
      helper.track = static_cast<o2::track::TrackParCov *>((AliExternalTrackParam *)track);
      int chargeIndex = track->GetSigned1Pt() > 0;
      helper.nSigmaTPC = mixTrack->nSigmaTPC;
      helper.nSigmaTOF = mixTrack->nSigmaTOF;
      helpers[fMixingTrack][chargeIndex].push_back(helper);
    }
  }

  fVertexer.setBz(esdEvent->GetMagneticField());
  fVertexerLambda.setBz(esdEvent->GetMagneticField());
  int indices[2][3]{{1, 1, 0}, {0, 0, 1}};

  std::unordered_map<int, int> mcMap;

  auto fillTreeInfo = [&](std::array<float, 3> nSigmaTPC, std::array<float, 3>, nSigmaTOF, float deuPhase) {
    lVector ldeu{(float)tracks[0]->Pt(), (float)tracks[0]->Eta(), (float)tracks[0]->Phi() + deuPhase, kDeuMass};
    lVector lpro{(float)tracks[1]->Pt(), (float)tracks[1]->Eta(), (float)tracks[1]->Phi(), kPMass};
    lVector lpi{(float)tracks[2]->Pt(), (float)tracks[2]->Eta(), (float)tracks[2]->Phi(), kPiMass};
    hypertriton = ldeu + lpro + lpi;
    o2RecHyp.mppi = (lpro + lpi).mass2();
    o2RecHyp.mdpi = (ldeu + lpi).mass2();

    ROOT::Math::Boost boostHyper{hypertriton.BoostToCM()};
    auto d{boostHyper(ldeu).Vect()};
    auto lambda{boostHyper(lpro + lpi).Vect()};
    auto p{boostHyper(lpro).Vect()};
    auto pi{boostHyper(lpi).Vect()};
    o2RecHyp.momDstar = std::sqrt(d.Mag2());
    o2RecHyp.cosThetaStar = d.Dot(hypertriton.Vect()) / (o2RecHyp.momDstar * hypertriton.P());
    o2RecHyp.cosTheta_ProtonPiH = p.Dot(pi) / std::sqrt(p.Mag2() * pi.Mag2());
    vert = fVertexer.getPCACandidate();
    decayVtx.SetCoordinates((float)(vert[0] - pvPos[0]), (float)(vert[1] - pvPos[1]), (float)(vert[2] - pvPos[2]));
    o2RecHyp.candidates = nVert;

    double deuPos[3], proPos[3], piPos[3];
    deuTrack.GetXYZ(deuPos);
    prTrack.GetXYZ(proPos);
    piTrack.GetXYZ(piPos);

    o2RecHyp.dca_de_pr = Hypot(deuPos[0] - proPos[0], deuPos[1] - proPos[1], deuPos[2] - proPos[2]);
    if (o2RecHyp.dca_de_pr > fMaxTrack2TrackDCA[0])
      continue;
    o2RecHyp.dca_de_pi = Hypot(deuPos[0] - piPos[0], deuPos[1] - piPos[1], deuPos[2] - piPos[2]);
    if (o2RecHyp.dca_de_pi > fMaxTrack2TrackDCA[1])
      continue;
    o2RecHyp.dca_pr_pi = Hypot(proPos[0] - piPos[0], proPos[1] - piPos[1], proPos[2] - piPos[2]);
    if (o2RecHyp.dca_pr_pi > fMaxTrack2TrackDCA[2])
      continue;

    o2RecHyp.dca_de_sv = Hypot(deuPos[0] - vert[0], deuPos[1] - vert[1], deuPos[2] - vert[2]);
    if (o2RecHyp.dca_de_sv > fMaxTrack2SVDCA[0])
      continue;
    o2RecHyp.dca_pr_sv = Hypot(proPos[0] - vert[0], proPos[1] - vert[1], proPos[2] - vert[2]);
    if (o2RecHyp.dca_pr_sv > fMaxTrack2SVDCA[1])
      continue;
    o2RecHyp.dca_pi_sv = Hypot(piPos[0] - vert[0], piPos[1] - vert[1], piPos[2] - vert[2]);
    if (o2RecHyp.dca_pi_sv > fMaxTrack2SVDCA[2])
      continue;

    o2RecHyp.chi2 = fVertexer.getChi2AtPCACandidate();

    const float mass = hypertriton.mass();
    if (mass < fMassWindow[0] || mass > fMassWindow[1])
      continue;

    const float totalMom = hypertriton.P();
    const float len = std::sqrt(decayVtx.Mag2());
    o2RecHyp->cosPA = hypertriton.Vect().Dot(decayVtx) / (totalMom * len);
    const float cosPA = fUseAbsCosPAcut ? std::abs(o2RecHyp->cosPA) : o2RecHyp->cosPA;
    o2RecHyp->ct = len * kHyperTritonMass / totalMom;
    if (o2RecHyp->ct < fCandidateCtRange[0] || o2RecHyp->ct > fCandidateCtRange[1])
      continue;
    if (fCosPAspline)
    {
      if (cosPA < fCosPAspline->Eval(o2RecHyp->ct))
        continue;
    }
    else if (cosPA < fMinCosPA)
    {
      continue;
    }
    o2RecHyp->r = decayVtx.Rho();
    o2RecHyp->positive = tracks[0]->Charge() > 0;
    o2RecHyp->pt = hypertriton.pt();
    o2RecHyp->phi = hypertriton.phi();
    o2RecHyp->pz = hypertriton.pz();
    o2RecHyp->m = mass;

    float dca[2], bCov[3];
    tracks[0]->GetImpactParameters(dca, bCov);
    o2RecHyp->dca_de = std::hypot(dca[0], dca[1]);
    tracks[1]->GetImpactParameters(dca, bCov);
    o2RecHyp->dca_pr = std::hypot(dca[0], dca[1]);
    pi.track->GetImpactParameters(dca, bCov);
    o2RecHyp->dca_pi = std::hypot(dca[0], dca[1]);

    o2RecHyp->hasTOF_de = HasTOF(tracks[0]);
    o2RecHyp->hasTOF_pr = HasTOF(tracks[1]);
    o2RecHyp->hasTOF_pi = HasTOF(tracks[2]);

    o2RecHyp->tofNsig_de = nSigmaTOF[0];
    o2RecHyp->tofNsig_pr = nSigmaTOF[1];
    o2RecHyp->tofNsig_pi = nSigmaTOF[2];

    o2RecHyp->tpcNsig_de = nSigmaTPC[0];
    o2RecHyp->tpcNsig_pr = nSigmaTPC[1];
    o2RecHyp->tpcNsig_pi = nSigmaTPC[2];

    o2RecHyp->tpcClus_de = tracks[0]->GetTPCsignalN();
    o2RecHyp->tpcClus_pr = tracks[1]->GetTPCsignalN();
    o2RecHyp->tpcClus_pi = tracks[2]->GetTPCsignalN();

    o2RecHyp->its_clusmap_de = tracks[0]->GetITSClusterMap();
    o2RecHyp->its_clusmap_pr = tracks[1]->GetITSClusterMap();
    o2RecHyp->its_clusmap_pi = tracks[2]->GetITSClusterMap();

    o2RecHyp->is_ITSrefit_de = tracks[0]->GetStatus() & AliVTrack::kITSrefit;
    o2RecHyp->is_ITSrefit_pr = tracks[1]->GetStatus() & AliVTrack::kITSrefit;
    o2RecHyp->is_ITSrefit_pi = tracks[2]->GetStatus() & AliVTrack::kITSrefit;

    if (fLambdaCheck)
    {
      int nVertLambda{0};
      try
      {
        nVertLambda = fVertexerLambda.process(*p.track, *pi.track);
      }
      catch (std::runtime_error &e)
      {
      }

      if (nVertLambda)
      {
        auto vertLambda = fVertexerLambda.getPCACandidate();
        fVertexerLambda.propagateTracksToVertex();
        auto &prTrackL = fVertexerLambda.getTrack(0);
        auto &piTrackL = fVertexerLambda.getTrack(1);
        ROOT::Math::XYZVectorF decayVtxLambda{(float)(vertLambda[0] - pvPos[0]), (float)(vertLambda[1] - pvPos[1]), (float)(vertLambda[2] - pvPos[2])};
        lVector lproL{(float)prTrackL.Pt(), (float)prTrackL.Eta(), (float)prTrackL.Phi(), kPMass};
        lVector lpiL{(float)piTrackL.Pt(), (float)piTrackL.Eta(), (float)piTrackL.Phi(), kPiMass};
        lVector lambda{lproL + lpiL};
        o2RecHyp->mppi_vert = lambda.mass();
        const float lambdaLen = std::sqrt(decayVtxLambda.Mag2());
        o2RecHyp->cosPA_Lambda = lambda.Vect().Dot(decayVtxLambda) / (lambda.P() * lambdaLen);
        o2RecHyp->dca_lambda_hyper = Hypot(vert[0] - vertLambda[0], vert[1] - vertLambda[1], vert[2] - vertLambda[2]);
      }
    }
    return true;
  };

  for (int idx{0}; idx < 2; ++idx)
  {
    for (const auto &deu : helpers[kDeuteron][indices[idx][0]])
    {

      for (const auto &p : helpers[kProton][indices[idx][1]])
      {
        if (deu.track == p.track)
          continue;

        for (const auto &pi : helpers[kPion][indices[idx][2]])
        {
          if (p.track == pi.track || deu.track == pi.track || deu.track == p.track)
            continue;

          lVector hypertriton;
          ROOT::Math::SVector<double, 3U> vert;
          ROOT::Math::XYZVectorF decayVtx;

          int nVert{0};
          try
          {
            nVert = fVertexer.process(*deu.track, *p.track, *pi.track);
          }
          catch (std::runtime_error &e)
          {
          }
          if (!nVert)
            continue;

          fVertexer.propagateTracksToVertex();
          auto &deuTrack = fVertexer.getTrack(0);
          auto &prTrack = fVertexer.getTrack(1);
          auto &piTrack = fVertexer.getTrack(2);

          std::array<AliESDtrack *, 3> tracks{(AliESDtrack *)deu.track, (AliESDtrack *)p.track, (AliESDtrack *)pi.track};
          std::array<float, 3> nSigmaTPC{deu.nSigmaTPC, p.nSigmaTPC, pi.nSigmaTPC};
          std::array<float, 3> nSigmaTOF{deu.nSigmaTOF, p.nSigmaTOF, pi.nSigmaTOF};

          if (fTrackRotations)
          {
            double step{TMath::TwoPi() / (fTrackRotations + 1)};
            for (int iR{1}; iR <= fTrackRotations; ++iR)
            {
              float deuPhase = iR * step;
              if (!fillTreeInfo(nSigmasTPC, nSigmasTOF, deuPhase))
                continue;
            }
          }

          else
          {
            if (!fillTreeInfo(nSigmasTPC, nSigmasTOF, 0))
              continue;
          }
          bool record{!fMC || !fOnlyTrueCandidates};
          if (fMC)
          {
            int momId = IsTrueHyperTriton3Candidate((AliESDtrack *)deu.track, (AliESDtrack *)p.track, (AliESDtrack *)pi.track, mcEvent);
            record = record || momId >= 0;
            if (record)
            {

              FillGenHypertriton(fGenHypO2, momId, true, mcEvent);
              mcMap[momId] = 1;
            }
          }
          if (record)
            fTreeHyp3->Fill();
        }
      }
    }
  }

  if (fMC)
  {
    RHyperTriton rec;
    rec.centrality = o2RecHyp->centrality;
    rec.trigger = o2RecHyp->trigger;
    *o2RecHyp = rec;
    for (int iTrack = 0; iTrack < mcEvent->GetNumberOfTracks(); iTrack++)
    {
      AliVParticle *part = mcEvent->GetTrack(iTrack);
      if (!part)
      {
        ::Warning("AliAnalysisTaskHypertriton3::UserExec",
                  "Generated loop %d - MC TParticle pointer to current stack particle = 0x0 ! Skipping.", iTrack);
        continue;
      }
      if (std::abs(part->Y()) > 1.)
        continue;
      if (!IsHyperTriton3(part, mcEvent))
        continue;
      if (mcMap.find(iTrack) != mcMap.end())
        continue;

      FillGenHypertriton(fGenHypO2, iTrack, false, mcEvent);
      fTreeHyp3->Fill();
    }
  }

  if (fEnableEventMixing)
  {
    FillEventMixingPool(fEventCuts.GetCentrality(), pvPos[2], deuteronsForMixing);
  }

  PostData(1, fListHist);
  PostData(2, fTreeHyp3);
}

void AliAnalysisTaskHypertriton3::Terminate(Option_t *) {}

int AliAnalysisTaskHypertriton3::FindEventMixingCentBin(const float centrality)
{
  if (centrality > 90)
    return -999;
  return static_cast<int>(centrality / 10);
}

int AliAnalysisTaskHypertriton3::FindEventMixingZBin(const float zvtx)
{
  if (zvtx > 10. || zvtx < -10.)
    return -999.;
  return static_cast<int>((zvtx + 10.) / 2);
}

void AliAnalysisTaskHypertriton3::FillEventMixingPool(const float centrality, const float zvtx,
                                                      const std::vector<EventMixingTrack> &tracks)
{
  int centBin = FindEventMixingCentBin(centrality);
  int zBin = FindEventMixingZBin(zvtx);

  auto &trackVector = fEventMixingPool[centBin][zBin];

  for (auto &t : tracks)
    trackVector.emplace_back(t);

  while (trackVector.size() > fEventMixingPoolDepth)
    trackVector.pop_front();

  return;
}

std::vector<EventMixingTrack *> AliAnalysisTaskHypertriton3::GetEventMixingTracks(const float centrality,
                                                                                  const float zvtx)
{
  int centBin = FindEventMixingCentBin(centrality);
  int zBin = FindEventMixingZBin(zvtx);

  std::vector<EventMixingTrack *> tmpVector;

  for (auto &v : fEventMixingPool[centBin][zBin])
  {
    if (v.used >= fEventMixingPoolMaxReuse)
      continue;
    tmpVector.emplace_back(&(v));
    v.used++;
  }

  return tmpVector;
}

AliAnalysisTaskHypertriton3 *AliAnalysisTaskHypertriton3::AddTask(bool isMC, TString suffix)
{
  // Get the current analysis manager
  AliAnalysisManager *mgr = AliAnalysisManager::GetAnalysisManager();
  if (!mgr)
  {
    ::Error("AddTaskHyperTriton2BodyML", "No analysis manager found.");
    return nullptr;
  }
  mgr->SetDebugLevel(2);

  // Check the analysis type using the event handlers connected to the analysis
  // manager.
  if (!mgr->GetInputEventHandler())
  {
    ::Error("AddTaskHypertritonO2", "This task requires an input event handler");
    return nullptr;
  }

  TString tskname = "AliAnalysisTaskHypertriton3";
  tskname.Append(suffix.Data());
  AliAnalysisTaskHypertriton3 *task = new AliAnalysisTaskHypertriton3(isMC, tskname.Data());

  AliAnalysisDataContainer *coutput1 = mgr->CreateContainer(
      Form("%s_summary", tskname.Data()), TList::Class(), AliAnalysisManager::kOutputContainer, "AnalysisResults.root");

  AliAnalysisDataContainer *coutput2 =
      mgr->CreateContainer(Form("HyperTritonTree%s", suffix.Data()), TTree::Class(),
                           AliAnalysisManager::kOutputContainer, Form("HyperTritonTree3.root:%s", suffix.Data()));
  coutput2->SetSpecialOutput();

  mgr->ConnectInput(task, 0, mgr->GetCommonInputContainer());
  mgr->ConnectOutput(task, 1, coutput1);
  mgr->ConnectOutput(task, 2, coutput2);
  return task;
}
