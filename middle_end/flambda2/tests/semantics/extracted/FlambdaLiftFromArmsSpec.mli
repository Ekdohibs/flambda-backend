open FlambdaHeapFreeSpec
open FlambdaLiftFromCodeSpec
open FlambdaLiftSpec
open FlambdaModel
open FlambdaWellFormedSpec

val lift_to_head : (path * lift_to) list -> expr -> expr

val lift_to_head_unit :
  unit_interface -> (path * lift_to) list -> expr -> expr option
