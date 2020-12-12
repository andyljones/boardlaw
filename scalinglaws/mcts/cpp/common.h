#include "../../cpp/common.h"

using F1D = TensorProxy<float, 1>;
using F2D = TensorProxy<float, 2>;
using F3D = TensorProxy<float, 3>;
using I1D = TensorProxy<int, 1>;
using I2D = TensorProxy<int, 2>;
using I3D = TensorProxy<int, 3>;
using B1D = TensorProxy<bool, 1>;
using B2D = TensorProxy<bool, 2>;

//TODO: Can I template-ize these classes?
struct MCTSPTA {
  F3D::PTA logits;
  F3D::PTA w; 
  I2D::PTA n; 
  F1D::PTA c_puct;
  I2D::PTA seats; 
  B2D::PTA terminal; 
  I3D::PTA children;
};

struct MCTS {
  F3D logits;
  F3D w; 
  I2D n; 
  F1D c_puct;
  I2D seats; 
  B2D terminal; 
  I3D children;

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
  I1D::PTA parents;
  I1D::PTA actions; 
};

struct Descent {
  I1D parents;
  I1D actions;

  DescentPTA pta() {
    return DescentPTA{
      parents.pta(),
      actions.pta()};
  }
};

struct BackupPTA {
  F3D::PTA v;
  F3D::PTA w;
  I2D::PTA n;
  F3D::PTA rewards;
  I2D::PTA parents;
  B2D::PTA terminal;
};

struct Backup {
  F3D v;
  F3D w;
  I2D n;
  F3D rewards;
  I2D parents;
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