#pragma once
#include "../../cpp/common.h"

namespace hexcuda {

TT step(TT board, TT seats, TT actions);

TT observe(TT board, TT seats);

}