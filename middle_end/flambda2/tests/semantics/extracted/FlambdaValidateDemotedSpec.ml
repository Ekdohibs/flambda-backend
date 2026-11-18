open Datatypes
open FlambdaDemoteSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaUnusedStaticsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateReadingSpec
open FlambdaValidateSpec

(** val after_demoted : expr -> expr **)

let after_demoted after = demote_handlers [] after

(** val validate_unit_demoted : nat -> dropped -> value_slot list -> loopified
    -> lifting list -> code_id list -> expr -> expr -> validation **)

let validate_unit_demoted rounds unused slots g ls kept before after =
  validate_unit rounds unused slots g ls kept before (after_demoted after)

(** val validate_unit_knowing_demoted : nat -> dropped -> value_slot list ->
    loopified -> lifting list -> code_id list -> expr -> expr -> validation **)

let validate_unit_knowing_demoted rounds unused slots g ls kept before after =
  validate_unit_knowing rounds unused slots g ls kept before
    (after_demoted after)

(** val validate_unit_reading_demoted : nat -> dropped -> value_slot list ->
    loopified -> lifting list -> code_id list -> expr -> expr -> validation **)

let validate_unit_reading_demoted rounds unused slots g ls kept before after =
  validate_unit_reading rounds unused slots g ls kept before
    (after_demoted after)

(** val validate_unit_knowing_reading_demoted : nat -> dropped -> value_slot
    list -> loopified -> lifting list -> code_id list -> expr -> expr ->
    validation **)

let validate_unit_knowing_reading_demoted rounds unused slots g ls kept before
    after =
  validate_unit_knowing_reading rounds unused slots g ls kept before
    (after_demoted after)
