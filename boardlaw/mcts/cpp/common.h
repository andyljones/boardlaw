#pragma once
#include "../../cpp/common.h"

//TODO: Can I template-ize these classes?
struct MCTSPTA {
  H3D::PTA logits;
  H3D::PTA w; 
  S2D::PTA n; 
  H1D::PTA c_puct;
  S2D::PTA seats; 
  B2D::PTA terminal; 
  S3D::PTA children;
};

struct MCTSTA {
  H3D::TA logits;
  H3D::TA w; 
  S2D::TA n; 
  H1D::TA c_puct;
  S2D::TA seats; 
  B2D::TA terminal; 
  S3D::TA children;
};

struct MCTS {
  H3D logits;
  H3D w; 
  S2D n; 
  H1D c_puct;
  S2D seats; 
  B2D terminal; 
  S3D children;

  MCTSPTA pta() {
    return MCTSPTA{
      logits.pta(), 
      w.pta(),
      n.pta(),
      c_puct.pta(),
      seats.pta(),
      terminal.pta(),
      children.pta()};
  }

  MCTSTA ta() {
    return MCTSTA{
      logits.ta(), 
      w.ta(),
      n.ta(),
      c_puct.ta(),
      seats.ta(),
      terminal.ta(),
      children.ta()};
  }
};

struct DescentPTA {
  S1D::PTA parents;
  S1D::PTA actions; 
};

struct DescentTA {
  S1D::TA parents;
  S1D::TA actions; 
};

struct Descent {
  S1D parents;
  S1D actions;

  DescentPTA pta() {
    return DescentPTA{
      parents.pta(),
      actions.pta()};
  }

  DescentTA ta() {
    return DescentTA{
      parents.ta(),
      actions.ta()};
  }

};

struct BackupPTA {
  H3D::PTA v;
  H3D::PTA w;
  S2D::PTA n;
  H3D::PTA rewards;
  S2D::PTA parents;
  B2D::PTA terminal;
};

struct BackupTA {
  H3D::TA v;
  H3D::TA w;
  S2D::TA n;
  H3D::TA rewards;
  S2D::TA parents;
  B2D::TA terminal;
};

struct Backup {
  H3D v;
  H3D w;
  S2D n;
  H3D rewards;
  S2D parents;
  B2D terminal;

  BackupPTA pta() {
    return BackupPTA{
      v.pta(),
      w.pta(),
      n.pta(),
      rewards.pta(),
      parents.pta(),
      terminal.pta()};
  }

  BackupTA ta() {
    return BackupTA{
      v.ta(),
      w.ta(),
      n.ta(),
      rewards.ta(),
      parents.ta(),
      terminal.ta()};
  }
};

namespace mctscuda {

Descent descend(MCTS m);
TT root(MCTS m);
void backup(Backup bk, TT leaves);

}