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


(* Proofs of the statement of FlambdaUnusedSlotsObservedSpec.v. Nothing
   here needs review.

   The simulation is that of FlambdaUnusedSlotsProofs.v, from a first
   state with a continuation environment of what may follow a unit.
   What may follow binds nothing static, so the rewrite leaves it as it
   is and the second state has the same continuations; and it projects
   none of the slots removed, which is what the invariant of the first
   run asks of the handlers that a state holds. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaClosuresProofs
  FlambdaImportsSpec FlambdaUnusedSlotsSpec FlambdaUnusedSlotsProofs
  FlambdaObserversSpec FlambdaUnitFromProofs FlambdaUnusedSlotsObservedSpec.
Import ListNotations.

(* A term with no static binding in it has no static set of closures:
   the rewrite leaves it as it is. *)
Fixpoint drop_slots_nothing (dropped : list value_slot) (e : expr)
    {struct e} :
  binds_no_static e = true -> drop_slots dropped e = e.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; reflexivity).
  - intros H. rewrite (drop_slots_nothing dropped body H). reflexivity.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros H.
    apply andb_true_iff in H as [Hh Hb].
    rewrite (drop_slots_nothing dropped handler Hh),
      (drop_slots_nothing dropped body Hb).
    reflexivity.
  - intros H. apply andb_true_iff in H as [Hhs Hb].
    rewrite (drop_slots_nothing dropped body Hb). f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhs. apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (drop_slots_nothing dropped handler Hh), (IH Hrest).
    reflexivity.
Qed.

(* What may follow a unit is the same in the second state. *)
Lemma observers_trimmed (dropped slots : list value_slot)
    (conts : cont_env) :
  observers slots conts -> trim_conts dropped conts = conts.
Proof.
  unfold observers, trim_conts.
  induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  rewrite (IH Hrest). f_equal. f_equal.
  revert Hc.
  destruct c
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; reflexivity ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (Hno & _ & _). simpl.
  rewrite (drop_slots_nothing dropped body Hno). reflexivity.
Qed.

(* And it projects no slot removed. *)
Lemma observers_slots_unused (dropped : list value_slot) (conts : cont_env) :
  observers dropped conts ->
  conts_terms_ok (handler_slots_unused dropped) conts.
Proof.
  unfold observers, conts_terms_ok.
  induction conts as [ | [k c] conts IH]; simpl; [ intros _; exact I | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  split; [ | exact (IH Hrest) ].
  revert Hc.
  destruct c
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; exact I ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (_ & _ & Hunused).
  split; [ exact Hunused | ]. split; [ reflexivity | exact I ].
Qed.

(* What [drop_unused_slots] gives refines the unit, the two run from
   what may follow a unit, from a static environment that holds no code
   and binds no closure with a slot removed, and from any heap. *)
Lemma drop_unused_slots_refines_under (dropped : list value_slot)
    (e e' : expr) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  drop_unused_slots dropped e = Some e' ->
  static_code statics = [] ->
  trim_statics dropped statics = statics ->
  observers dropped conts ->
  refines (State e [] conts statics hp rd) (State e' [] conts statics hp rd).
Proof.
  unfold drop_unused_slots.
  destruct (slots_unused dropped e) eqn:Hunused; [ | discriminate ].
  intros [= <-] Hno_code Htrim Hobservers.
  apply (simulation_refines (dropped_related dropped)
           (dropped_related_step dropped)).
  split.
  - unfold trim_state. simpl_state.
    rewrite Htrim, (observers_trimmed dropped dropped conts Hobservers).
    reflexivity.
  - unfold slots_unused_inv, terms_inv. simpl_state.
    refine
      (conj Hunused
         (conj (observers_slots_unused dropped conts Hobservers) _)).
    intros id c Hlookup. rewrite Hno_code in Hlookup. discriminate Hlookup.
Qed.

Theorem drop_unused_slots_refines_observed :
  drop_unused_slots_refines_observed_statement.
Proof.
  intros imports dropped conts e e' Hdrop Hobservers.
  exact (drop_unused_slots_refines_under dropped e e' conts
           (imports_statics imports) empty_heap no_reads Hdrop eq_refl
           (trim_statics_imports dropped imports) Hobservers).
Qed.

(* So the two runs come to results that differ in nothing that an exit
   is compared by, from every continuation environment of what may
   follow a unit that projects no slot removed. *)
Lemma drop_unused_slots_observed (imports : list (symbol * import_class))
    (dropped : list value_slot) (e e' : expr) :
  drop_unused_slots dropped e = Some e' ->
  refines_under_observers dropped (imports_statics imports) empty_heap no_reads
    e e'.
Proof.
  intros Hdrop conts Hobservers. apply refines_observed.
  exact (drop_unused_slots_refines_observed imports dropped conts e e' Hdrop
           Hobservers).
Qed.
