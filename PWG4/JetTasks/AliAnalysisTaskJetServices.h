#ifndef ALIANALYSISTASKJETSERVICES_H
#define ALIANALYSISTASKJETSERVICES_H
 
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

// **************************************
// Task used for the correction of determiantion of reconstructed jet spectra
// Compares input (gen) and output (rec) jets   
// *******************************************

#include  "AliAnalysisTaskSE.h"
#include  "THnSparse.h" // cannot forward declare ThnSparseF

////////////////
class AliJetHeader;
class AliESDEvent;
class AliESDVertex;
class AliAODEvent;
class AliAODVertex;
class AliAODJet;
class AliGenPythiaEventHeader;
class AliCFManager;
class AliTriggerAnalysis;

class TList;
class TChain;
class TH1F;
class TH2F;
class TH3F;
class TProfile;



class AliAnalysisTaskJetServices : public AliAnalysisTaskSE
{
 public:
    AliAnalysisTaskJetServices();
    AliAnalysisTaskJetServices(const char* name);
    virtual ~AliAnalysisTaskJetServices() {;}
    // Implementation of interface methods
    virtual void UserCreateOutputObjects();
    virtual void Init();
    virtual void LocalInit() { Init(); }
    virtual void UserExec(Option_t *option);
    virtual void Terminate(Option_t *option);
    virtual void SetZVertexCut(Float_t f){fVtxZCut = f;}
    virtual void SetPtMinCosmic(Float_t ptMin) {fPtMinCosmic = ptMin;}
    virtual void SetRMinCosmic(Float_t rMin) {fRIsolMinCosmic = rMin;}
    virtual void SetMaxCosmicAngle(Float_t angle)   {fMaxCosmicAngle = angle;}
    virtual Bool_t Notify();

    virtual void SetAODInput(Bool_t b){fUseAODInput = b;}
    virtual void SetRunRange(Float_t fLo,Float_t fUp){fRunRange[0] = fLo;fRunRange[1] = fUp;}
    virtual void SetMCData(Bool_t b){fMC = b;}
    virtual void SetUsePhysicsSelection(Bool_t b){fUsePhysicsSelection = b;}
    virtual void SetPhysicsSelectionFlag(Int_t i){fPhysicsSelectionFlag = i;}
    Bool_t IsEventSelected(const AliESDEvent* esd);
    Bool_t IsEventSelected(const AliAODEvent* aod) const;

    Bool_t IsEventPileUp(const AliESDEvent* esd) const;

    Bool_t IsEventCosmic(const AliESDEvent* esd) const;

    Bool_t IsVertexValid(const AliESDVertex *vtx);
    Bool_t IsVertexValid(const AliAODVertex *vtx) const;

    Bool_t IsVertexIn(const AliESDVertex *vtx);
    Bool_t IsVertexIn(const AliAODVertex *vtx) const;

    enum { kAllTriggered = 0,kTriggeredVertex,kTriggeredVertexIn,kSelectedALICE,kSelectedALICEVertexValid,kSelectedALICEVertexIn,kSelected,kConstraints};

    enum { kNoEventCut=1<<0,
	   kPhysicsSelectionCut=1<<1,
	   kContributorsCut1=1<<2,
	   kContributorsCut2=1<<3,
	   kContributorsCut3=1<<4,
	   kVertexTPC=1<<5,
	   kVertexSPD=1<<6,
	   kVertexGlobal=1<<7,
	   kSPDDispersionCut=1<<8,
	   kVertexZCut=1<<9,
	   kVertexRCut=1<<10,
	   kTotalEventCuts=(1<<11)-1};

 private:

    AliAnalysisTaskJetServices(const AliAnalysisTaskJetServices&);
    AliAnalysisTaskJetServices& operator=(const AliAnalysisTaskJetServices&);

    Bool_t        fUseAODInput;        // take jet from input AOD not from ouptu AOD
    Bool_t        fUsePhysicsSelection;// decide wether we take into account physicsselction task
    Bool_t        fMC;                 // true for MC data to allow correct trigger slection
    UInt_t        fPhysicsSelectionFlag; // defines the glag for acceptance of events from physics selection
    UInt_t        fSelectionInfoESD;   // slection info bit mask
    UInt_t        fEventCutInfoESD;   // event selection info of what is cutted after physics selection
    Float_t       fAvgTrials;          // Average number of trials
    Float_t       fVtxXMean;           // mean x for cuts
    Float_t       fVtxYMean;           // mean y for cuts
    Float_t       fVtxZMean;           // mean z for cuts
    Float_t       fVtxRCut;            // vtx cut in R
    Float_t       fVtxZCut;            // vtzx cut in Z
    Float_t       fPtMinCosmic;        // Minimum pT to be considered as cosmic candidate
    Float_t       fRIsolMinCosmic;     // Minimum R = sqrt{deltaPhi^2 + deltaEta^2} to be considered as cosmic candidate
    Float_t       fMaxCosmicAngle;     // Max deviation from pi (angle between two tracks) in case of cosmic candidate
    Float_t       fRunRange[2];        // only important for real data for 
    TProfile*     fh1Xsec;             //! pythia cross section and trials
    TH1F*         fh1Trials;           //! trials are added
    TH1F*         fh1PtHard;           //! Pt har of the event...       
    TH1F*         fh1PtHardTrials;     //! Number of trials 
    TH1F*         fh1SelectionInfoESD; //! Masks that satisfy fSelectionInfo
    TH1F*         fh1EventCutInfoESD;  //! Masks that satisfy fSelectionInfo
    TH2F*         fh2TriggerCount;     //! number of fire triggers in each case
    TH2F*         fh2ESDTriggerCount;  //! number of fire triggers in each case
    TH2F*         fh2TriggerVtx;       //! vtx. position vs. trigger decision
    TH2F*         fh2ESDTriggerVtx;    //! vtx. position vs. trigger decision 
    TH2F*         fh2ESDTriggerRun;    //! fired triggers vs. run number
    TH2F*         fh2VtxXY;            //! XY position of VTX were available
    TH1F*         fh1NCosmicsPerEvent;  //! Number of coscmic candidates found in event
    AliTriggerAnalysis *fTriggerAnalysis; //! Trigger Analysis to get the background rates etc.

    TList *fHistList; // Output list
   
    ClassDef(AliAnalysisTaskJetServices,9)
};
 
#endif
