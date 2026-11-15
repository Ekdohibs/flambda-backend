(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)


(* The proofs of FlambdaExitBoundSpec.v: a refinement, with the theorem
   on the static names that a well-formed unit holds when it runs
   (FlambdaStaticsInScopeProofs.statics_bound_in_run), gives the
   refinement with the exits related by [same_exit_bound]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaHeapSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaExitWorldSpec FlambdaExitWorldProofs
  FlambdaWellFormedSpec
  FlambdaStaticsInScopeSpec FlambdaStaticsInScopeProofs
  FlambdaHoistStaticsProofs FlambdaUnusedStaticsSpec FlambdaLoopifySpec
  FlambdaLiftSpec FlambdaValidateSpec FlambdaValidateProofs
  FlambdaImportsSpec FlambdaExitBoundSpec.
Import ListNotations.

(* The static environment that a unit is run from binds its imports and
   nothing else, to things that mention nothing. *)
Lemma imports_statics_bound (imports : list (symbol * import_class)) :
  imports_bound (map fst imports) (imports_statics imports).
Proof.
  split; [reflexivity|].
  split; [|exact (imports_statics_among _ imports)].
  intros s.
  unfold symbol_bound.
  change (is_bound s (static_values (imports_statics imports))
          = among s (map fst imports)).
  rewrite is_bound_among.
  unfold imports_statics. cbn [static_values].
  rewrite map_map.
  f_equal.
  apply map_ext.
  intros [s' c]. reflexivity.
Qed.

Theorem unit_refines_exits_bound : unit_refines_exits_bound_statement.
Proof.
  intros imports u source target Hwf Hscope Hconts Href.
  apply (proj2 (refines_world_same _ _)) in Href.
  destruct Href as [Hexits Hends].
  split; [|exact Hends].
  intros w Hw.
  destruct (Hexits w Hw) as [w' [Hw' Hsame]].
  exists w'.
  split; [exact Hw'|].
  split; [exact Hsame|].
  destruct Hw' as [n [last [Hrun [Hstep Hstatics]]]].
  destruct
    (statics_bound_in_run (map fst imports) u target
       (imports_statics imports) empty_heap no_reads n last Hwf Hscope
       Hconts (imports_statics_bound imports) (heap_values_all_empty _)
       Hrun)
    as [Hbound [_ Hexit]].
  rewrite Hstep in Hexit.
  unfold exit_statics_bound.
  rewrite <- Hstatics.
  split; [exact Hexit|].
  rewrite (step_exit_heap last _ _ _ _ Hstep).
  exact (proj2 (proj2 (proj2 Hbound))).
Qed.

Theorem validated_exits_bound : validated_exits_bound_statement.
Proof.
  intros u rounds unused slots G Ls kept before after Hverdict Hwf Hscope
    Hconts.
  apply (unit_refines_exits_bound [] u before after Hwf Hscope Hconts).
  exact (validate_unit_sound rounds unused slots G Ls kept before after
           Hverdict).
Qed.

(* ======================================================================== *)
(* Two units of which the second exits with a symbol that it has not bound  *)
(* ======================================================================== *)

(* Two states that exit alike: the second refines the first. *)
Lemma exits_alike_refines (s t : state) (k : continuation)
    (vs : list value) (hp : heap) :
  behaves s (Exits k vs hp) -> behaves t (Exits k vs hp) -> refines s t.
Proof.
  intros Hs Ht.
  split.
  - intros k' vs' hp' Hexit.
    rewrite (behaviour_unique s _ _ Hexit Hs).
    exact Ht.
  - intros b _ Hb.
    left.
    rewrite (behaviour_unique t _ _ Hb Ht).
    exact Hs.
Qed.

(* The return continuation of the two units is [1] and their exception
   continuation [2]. The symbol is [1]. *)
Definition exit_interface : unit_interface :=
  Unit_interface 1%positive 2%positive.

Definition exit_block : static_const :=
  Static_const_block 0 [Const (Tagged_immediate 7)].

(* The first unit binds the symbol and returns it. *)
Definition bound_exit_unit : expr :=
  Let_static [Bind_symbol 1%positive exit_block]
    (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None)).

(* The second: the variable of the test is [2]. The handler of [4]
   binds the symbol and returns it; the handler of [3] returns the
   symbol; and the test goes to [3]. *)
Definition unbound_exit_unit : expr :=
  Let 2%positive (Naked_number Kind_naked_immediate)
    (Simple (Const (Naked_immediate 1)))
    (Let_cont_non_recursive 4%positive
       (Handler [] (make_handler_metadata [])
          (Let_static [Bind_symbol 1%positive exit_block]
             (Apply_cont
                (Apply_cont_expr 1%positive [Symbol 1%positive] None))))
       (Let_cont_non_recursive 3%positive
          (Handler [] (make_handler_metadata [])
             (Apply_cont
                (Apply_cont_expr 1%positive [Symbol 1%positive] None)))
          (Switch (Var 2%positive)
             [(0%Z, Apply_cont_expr 4%positive [] None);
              (1%Z, Apply_cont_expr 3%positive [] None)]))).

(* The state of the second unit whose step is the exit. *)
Definition unbound_exit_state : state :=
  match
    run 5
      (State unbound_exit_unit [] [] empty_static_env empty_heap no_reads)
  with
  | Step s => s
  | Exit _ _ _ _ | Undefined =>
    State Invalid [] [] empty_static_env empty_heap no_reads
  end.

Theorem refined_exit_unbound : refined_exit_unbound_statement.
Proof.
  assert (Hsource :
    behaves (State bound_exit_unit [] [] empty_static_env empty_heap no_reads)
      (Exits 1%positive [Value_symbol 1%positive] empty_heap)).
  { exists 2%nat, no_reads. vm_compute. reflexivity. }
  assert (Htarget :
    behaves
      (State unbound_exit_unit [] [] empty_static_env empty_heap no_reads)
      (Exits 1%positive [Value_symbol 1%positive] empty_heap)).
  { exists 6%nat, no_reads. vm_compute. reflexivity. }
  exists exit_interface, bound_exit_unit, unbound_exit_unit, 5%nat,
    unbound_exit_state, 1%positive, 1%positive.
  split; [vm_compute; reflexivity|].
  split; [vm_compute; reflexivity|].
  split; [vm_compute; reflexivity|].
  split; [exact Hsource|].
  split; [exact (exits_alike_refines _ _ _ _ _ Hsource Htarget)|].
  repeat split; vm_compute; reflexivity.
Qed.
