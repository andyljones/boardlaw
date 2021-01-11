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
};

struct DescentPTA {
  S1D::PTA parents;
  S1D::PTA actions; 
};

struct Descent {
  S1D parents;
  S1D actions;

  DescentPTA pta() {
    return DescentPTA{
      parents.pta(),
      actions.pta()};
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
};

Descent descend(MCTS m);
TT root(MCTS m);
void backup(Backup m, TT leaves);