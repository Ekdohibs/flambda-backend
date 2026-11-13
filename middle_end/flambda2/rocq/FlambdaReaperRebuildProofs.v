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

(* The proofs of FlambdaReaperRebuildSpec.v. Nothing here needs review.

   The proof of [rebuild_refines] is a simulation of the run of the unit
   by the run of the rebuilt unit, in which a step of the first is
   answered by one step of the second, or by none where the rebuild
   removed a [Let] or a [Let_cont] (the expression of the first then
   shrinks), or by anything where the first goes wrong in the next step
   ([shrinking_refines]). The relation ([related]) has:
   - the invariant of FlambdaReaperFlowProofs.v on the first run
     ([flow_invariant]): where the values of the first run may be;
   - the facts of the expression of the first run in the graph
     ([expr_edges]), at toplevel, under the nodes of the parameters of
     the continuations in scope, which the relation of the continuations
     says are those of the handlers bound (the invariant of the flow has
     the facts too, but allows the nodes of the continuations of code
     being run, which are not those of the handlers);
   - an environment of the rebuild ([rebuild_env]) under which the
     expression of the second is the rebuild of that of the first;
   - the variables of the first run that the rebuild keeps bound in the
     second to related values ([env_ok], [rel]): the same constants and
     symbols, a block with the same tag and as many fields, each field
     that is used related, a boxed number or poison where its contents
     is not used, an array of elements related at nodes with any usage,
     and a closure of the same set with the value slots that the rebuild
     keeps, each used one related. [rel] says which allocation the value
     may come from, as [flows] does, so that the fields kept are known;
   - the continuations that the expression of the second names, related
     in the two runs ([conts_related], [cont_rel]): both ways out of the
     term through a continuation of the unit, or handlers, the second
     the rebuild of the first, with the mask of the parameters kept;
   - the static environments related ([statics_rel]), as the
     environments of variables are, for the symbols that have a use;
   - the same heap, and what the unit asks of the expression of the first
     ([expr_ok]): its [Switch]es are on what has any usage, it gives the
     continuations of the unit one value, and it meets the check
     ([checked]: [no_code_no_slots] if [exact], [no_code] otherwise);
   - the projections of a set that the rebuild has pending
     ([pending_ok]).
   At a node with any usage, what the second run has for a value is the
   same but for the value slots of its closures, and the same if [exact]
   ([sim], [rel_au_sim]): an exit is observed equal in the two runs, and
   equal if [exact] ([exits_related]). A primitive of the default case
   sees no value slot ([sim_respects], with
   FlambdaPrimitiveSpec.eval_prim_related).

   The facts of the graph that the proof reads are those of
   FlambdaReaperFlowProofs.expr_edges, and one more: the scrutinee of a
   [Switch] has any usage ([traverse_scrutinees]), so that the rebuild
   keeps it. The rebuild keeps every variable of kind [Region] or
   [Rec_info] ([is_var_used]), used or not; in the fragment such a
   variable holds a constant, given by a simple or by the identity, the
   same in the two runs ([region_rel]).

   The two statements are the two modes of one proof
   ([rebuild_refines_checked]): [rebuild_refines] with [exact], for
   which a closure has no value slot, and [rebuild_refines_observably]
   without. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaUnusedStaticsSpec FlambdaBindersSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaReaperRebuild FlambdaReaperRebuildSpec.
Import ListNotations.

(* ======================================================================== *)
(* Simulations in which a step may be answered by none                      *)
(* ======================================================================== *)

Section Shrinking.

Variable R : state -> state -> Prop.
Variable E : list value -> list value -> Prop.
Variable measure : state -> nat.

(* A step of the first state is answered by the second: nothing is asked
   if it is undefined, or if the state it comes to is; an exit, by an
   exit in one step through the same continuation, with the same heap
   and values related by [E]; a step, by one step to a related state,
   or by none, to a state related to the second, if the measure goes
   down. *)
Definition answered (s t : state) : Prop :=
  match step s with
  | Undefined => True
  | Exit k vs hp => ∃ vs' : list value, step t = Exit k vs' hp ∧ E vs vs'
  | Step s' =>
    step s' = Undefined
    ∨ (∃ t' : state, step t = Step t' ∧ R s' t')
    ∨ (measure s' < measure s ∧ R s' t)
  end.

Hypothesis R_answered : ∀ s t : state, R s t -> answered s t.

(* An exit of the first run is one of the second, with related values. *)
Lemma shrinking_exit (n : nat) :
  ∀ (s t : state) (k : continuation) (vs : list value) (hp : heap),
    R s t -> run n s = Exit k vs hp ->
    ∃ (m : nat) (vs' : list value), run m t = Exit k vs' hp ∧ E vs vs'.
Proof.
  induction n as [ | n IH]; intros s t k vs hp Hrel Hrun;
    [ discriminate Hrun | ].
  rewrite run_succ in Hrun. pose proof (R_answered s t Hrel) as Ha.
  unfold answered in Ha.
  destruct (step s) as [s' | k' vs' hp' | ] eqn:Hs.
  - destruct Ha as [Hund | [(t' & Ht & Hrel') | (_ & Hrel')]].
    + destruct n as [ | n]; [ discriminate Hrun | ].
      rewrite run_succ, Hund in Hrun. discriminate Hrun.
    + destruct (IH s' t' k vs hp Hrel' Hrun) as (m & vs2 & Hm & HE).
      exists (S m), vs2. rewrite run_succ, Ht. split; [ exact Hm | exact HE ].
    + exact (IH s' t k vs hp Hrel' Hrun).
  - injection Hrun as <- <- <-. destruct Ha as (vs2 & Ha & HE).
    exists 1, vs2. rewrite run_succ, Ha. split; [ reflexivity | exact HE ].
  - discriminate Hrun.
Qed.

(* Behaviours of the two runs that correspond: exits through the same
   continuation, with the same heap and values related by [E]; or
   both going wrong. *)
Definition behaviour_related (b b' : behaviour) : Prop :=
  match b, b' with
  | Exits k vs hp, Exits k' vs' hp' => k = k' ∧ hp = hp' ∧ E vs vs'
  | Goes_wrong, Goes_wrong => True
  | _, _ => False
  end.

(* If the second run has ended, the first ends in a corresponding way or
   goes wrong: each step of the first uses up one of the second, or
   lowers the measure. *)
Lemma shrinking_ends (m : nat) :
  ∀ (s t : state) (b : behaviour),
    R s t -> ended (run m t) b ->
    (∃ b0 : behaviour, behaves s b0 ∧ behaviour_related b0 b)
    ∨ behaves s Goes_wrong.
Proof.
  induction m as [m IHm] using (well_founded_induction lt_wf).
  intros s. remember (measure s) as k eqn:Hk. revert s Hk.
  induction k as [k IHk] using (well_founded_induction lt_wf).
  intros s Hk t b Hrel Hended.
  pose proof (R_answered s t Hrel) as Ha. unfold answered in Ha.
  destruct (step s) as [s' | k' vs hp | ] eqn:Hs.
  - destruct Ha as [Hund | [(t' & Ht & Hrel') | (Hlt & Hrel')]].
    + right. exists 2. rewrite run_succ, Hs, run_succ, Hund. reflexivity.
    + destruct m as [ | m]; [ destruct b; contradiction | ].
      rewrite run_succ, Ht in Hended.
      destruct (IHm m (Nat.lt_succ_diag_r m) s' t' b Hrel' Hended)
        as [(b0 & Hb & Hbr) | Hw];
        [ left; exists b0; split; [ | exact Hbr ] | right ];
        apply step_behaves; rewrite Hs; assumption.
    + subst k.
      destruct (IHk (measure s') Hlt s' eq_refl t b Hrel' Hended)
        as [(b0 & Hb & Hbr) | Hw];
        [ left; exists b0; split; [ | exact Hbr ] | right ];
        apply step_behaves; rewrite Hs; assumption.
  - destruct Ha as (vs' & Ha & HE).
    destruct m as [ | m]; [ destruct b; contradiction | ].
    rewrite run_succ, Ha in Hended. apply ended_exit in Hended. subst b.
    left. exists (Exits k' vs hp). split.
    + apply step_behaves. rewrite Hs. reflexivity.
    + split; [ reflexivity | split; [ reflexivity | exact HE ] ].
  - right. apply step_behaves. rewrite Hs. reflexivity.
Qed.

Lemma shrinking_refines (s t : state) :
  (∀ vs vs' : list value, E vs vs' -> vs' = vs) -> R s t -> refines s t.
Proof.
  intros HE Hrel. split.
  - intros k vs hp [n Hn].
    destruct (shrinking_exit n s t k vs hp Hrel Hn) as (m & vs' & Hm & Hvs).
    rewrite (HE _ _ Hvs) in Hm. exists m. exact Hm.
  - intros b Hterm Hb. destruct (behaves_ended t b Hterm Hb) as [m Hm].
    destruct (shrinking_ends m s t b Hrel Hm) as [(b0 & Hb0 & Hbr) | Hw];
      [ left | right; exact Hw ].
    destruct b0 as [k vs hp | | ]; destruct b as [k' vs' hp' | | ];
      try contradiction; [ | exact Hb0 ].
    destruct Hbr as (<- & <- & Hvs). rewrite (HE _ _ Hvs). exact Hb0.
Qed.

Lemma shrinking_refines_observably (s t : state) :
  (∀ vs vs' : list value, E vs vs' -> values_related observed_equal vs vs') ->
  R s t -> refines_observably s t.
Proof.
  intros HE Hrel. split; [ | split ].
  - intros k vs hp [n Hn].
    destruct (shrinking_exit n s t k vs hp Hrel Hn) as (m & vs' & Hm & Hvs).
    exists vs'. split; [ exists m; exact Hm | exact (HE _ _ Hvs) ].
  - intros k vs' hp Hb.
    destruct (behaves_ended t (Exits k vs' hp) ltac:(discriminate) Hb)
      as [m Hm].
    destruct (shrinking_ends m s t _ Hrel Hm) as [(b0 & Hb0 & Hbr) | Hw];
      [ left | right; exact Hw ].
    destruct b0 as [k0 vs hp0 | | ]; try contradiction.
    destruct Hbr as (-> & -> & Hvs). exists vs.
    split; [ exact Hb0 | exact (HE _ _ Hvs) ].
  - intros Hb.
    destruct (behaves_ended t Goes_wrong ltac:(discriminate) Hb) as [m Hm].
    destruct (shrinking_ends m s t _ Hrel Hm) as [(b0 & Hb0 & Hbr) | Hw];
      [ | exact Hw ].
    destruct b0; try contradiction. exact Hb0.
Qed.

End Shrinking.

Lemma values_related_refl (l : list value) :
  Forall (fun v => observed_equal v v) l -> values_related observed_equal l l.
Proof.
  induction 1 as [ | v l Hv _ IH]; simpl; [ exact I | split; assumption ].
Qed.

Lemma observed_equal_refl : ∀ v : value, observed_equal v v.
Proof.
  apply FlambdaHeapProofs.value_induction; cbn [observed_equal].
  - reflexivity.
  - reflexivity.
  - split; reflexivity.
  - reflexivity.
  - intros tag fields IH. split; [ reflexivity | ].
    exact (values_related_refl fields IH).
  - intros elements IH. exact (values_related_refl elements IH).
  - reflexivity.
  - reflexivity.
  - intros callee args arity Hcallee IH.
    split; [ exact Hcallee | split; [ exact (values_related_refl args IH)
                                    | reflexivity ] ].
Qed.

Lemma values_observed_refl (l : list value) :
  values_related observed_equal l l.
Proof.
  apply values_related_refl, Forall_forall. intros v _.
  apply observed_equal_refl.
Qed.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

(* [P i v v'] of the elements at each position of two lists of one
   length, the first at [i]. *)
Definition fields_related (P : nat -> value -> value -> Prop)
    : nat -> list value -> list value -> Prop :=
  fix fields_related (i : nat) (fields fields' : list value)
      {struct fields} : Prop :=
    match fields, fields' with
    | [], [] => True
    | field :: fields, field' :: fields' =>
      P i field field' ∧ fields_related (S i) fields fields'
    | _, _ => False
    end.

Lemma fields_related_nth (P : nat -> value -> value -> Prop) (k : nat)
    (l l' : list value) (i : nat) (v : value) :
  fields_related P k l l' -> nth_error l i = Some v ->
  ∃ v' : value, nth_error l' i = Some v' ∧ P (k + i) v v'.
Proof.
  revert k l' i. induction l as [ | w l IH]; intros k [ | w' l'] i H Hnth;
    simpl in H; try contradiction; [ destruct i; discriminate | ].
  destruct H as [Hw Hl]. destruct i as [ | i]; simpl in Hnth |- *.
  - injection Hnth as <-. exists w'. rewrite Nat.add_0_r. auto.
  - replace (k + S i) with (S k + i) by lia. exact (IH _ _ _ Hl Hnth).
Qed.

Lemma fields_related_nth' (P : nat -> value -> value -> Prop) (k : nat)
    (l l' : list value) (i : nat) (v' : value) :
  fields_related P k l l' -> nth_error l' i = Some v' ->
  ∃ v : value, nth_error l i = Some v ∧ P (k + i) v v'.
Proof.
  revert k l' i. induction l as [ | w l IH]; intros k [ | w' l'] i H Hnth;
    simpl in H; try contradiction; [ destruct i; discriminate | ].
  destruct H as [Hw Hl]. destruct i as [ | i]; simpl in Hnth |- *.
  - injection Hnth as <-. exists w. rewrite Nat.add_0_r. auto.
  - replace (k + S i) with (S k + i) by lia. exact (IH _ _ _ Hl Hnth).
Qed.

Lemma fields_related_length (P : nat -> value -> value -> Prop) (k : nat)
    (l l' : list value) :
  fields_related P k l l' -> length l' = length l.
Proof.
  revert k l'. induction l as [ | w l IH]; intros k [ | w' l'] H;
    simpl in H; try contradiction; [ reflexivity | ].
  simpl. f_equal. exact (IH _ _ (proj2 H)).
Qed.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

(* Induction on values, with the fields of a block. *)
Lemma value_block_ind (P : value -> Prop) :
  (∀ (tag : Z) (fields : list value),
     (∀ field : value, field ∈ fields -> P field) ->
     P (Value_block tag fields)) ->
  (∀ v : value, (∀ (tag : Z) (fields : list value),
                   v <> Value_block tag fields) -> P v) ->
  ∀ v : value, P v.
Proof.
  intros Hblock Hother. fix IH 1. intros v.
  destruct v as [c | code vs | slot decls vs | s | tag fields | elements | c
                | l | callee args arity];
    try (apply Hother; intros tag' fields' Heq; discriminate Heq).
  apply Hblock. revert fields. fix go 1.
  intros [ | field fields] field' Hin; [ destruct Hin | ].
  destruct Hin as [<- | Hin];
    [ exact (IH field) | exact (go fields field' Hin) ].
Qed.

(* [P w x] of each binding [(w, x)] of a list of value slots that no
   earlier binding shadows: of what [lookup] finds. *)
Definition slots_related (P : value_slot -> value -> Prop)
    : list (value_slot * value) -> list value_slot -> Prop :=
  fix slots_related (vs : list (value_slot * value))
      (seen : list value_slot) {struct vs} : Prop :=
    match vs with
    | [] => True
    | (w, x) :: vs =>
      (among w seen = false -> P w x) ∧ slots_related vs (w :: seen)
    end.

Lemma slots_related_iff (P : value_slot -> value -> Prop)
    (vs : list (value_slot * value)) :
  slots_related P vs []
  <-> ∀ (w : value_slot) (x : value), lookup w vs = Some x -> P w x.
Proof.
  enough (H : ∀ seen : list value_slot,
            slots_related P vs seen
            <-> ∀ (w : value_slot) (x : value),
                  among w seen = false -> lookup w vs = Some x -> P w x)
    by (rewrite H; split; intros H' w x; [ apply H'; reflexivity | ];
        intros _; apply H').
  induction vs as [ | [w' x'] vs IH]; intros seen; simpl.
  - split; [ intros _ w x _ Hl; discriminate Hl | intros _; exact I ].
  - rewrite IH. split.
    + intros [Hw Hrest] w x Hseen Hl.
      destruct (Pos.eqb_spec w w') as [-> | Hne].
      * injection Hl as <-. exact (Hw Hseen).
      * apply (Hrest w x); [ | exact Hl ].
        unfold among in Hseen |- *. simpl.
        rewrite (proj2 (Pos.eqb_neq w w') Hne). exact Hseen.
    + intros H. split.
      * intros Hseen. apply (H w' x' Hseen). rewrite Pos.eqb_refl.
        reflexivity.
      * intros w x Hseen Hl. unfold among in Hseen. simpl in Hseen.
        apply orb_false_iff in Hseen as [Hne Hseen].
        apply (H w x Hseen). rewrite Hne. exact Hl.
Qed.

(* Leaves the cases of [rel] that are not in the fragment. *)
Ltac not_rel :=
  try (match goal with |- False -> _ => intros [] end).

Section Rel.

Variable G : graph.
Variable sol : solution.
Variable kept : value_slot -> bool.
Variable exact : Prop.

(* What the second run has for a value of the first at a node with any
   usage: the same, but for the value slots of the closures in it, which
   are the same if [exact]; each symbol in it has any usage. *)
Fixpoint sim (v v' : value) {struct v} : Prop :=
  match v with
  | Value_const _ | Value_boxed_number _ | Value_partial_application _ _ _ =>
    v' = v
  | Value_symbol s => v' = v ∧ s_any_usage sol (Node_symbol s) = true
  | Value_block tag fields =>
    ∃ fields' : list value,
      v' = Value_block tag fields'
      ∧ values_related (fun w w' => sim w w') fields fields'
  | Value_immutable_array elements =>
    ∃ elements' : list value,
      v' = Value_immutable_array elements'
      ∧ values_related (fun w w' => sim w w') elements elements'
  | Value_closure_of_set slot function_decls value_slots =>
    ∃ value_slots' : list (value_slot * value),
      v' = Value_closure_of_set slot function_decls value_slots'
      ∧ (exact -> value_slots' = value_slots)
  | Value_closure _ _ | Value_location _ => False
  end.

(* The value [v] of the first run may be at the node [n], and [v'] is
   what the second run has for it. A closure has the nodes of the
   closures of its set ([sites]) as FlambdaReaperFlowSpec.flows has them;
   in the second run it is of the same set, with the value slots that the
   rebuild keeps ([exact]: all), and a value slot kept and used at one of
   these nodes is related there. The elements of an immutable array are
   related at nodes with any usage. *)
Fixpoint rel (v : value) (n : node) (v' : value) {struct v} : Prop :=
  match v with
  | Value_const _ => s_any_source sol n = true ∧ v' = v
  | Value_block tag fields =>
    ∃ (a : node) (fields' : list value),
      may_come_from sol a n
      ∧ v' = Value_block tag fields'
      ∧ fields_related
          (fun i field field' =>
             ∃ from : node,
               g_constructor G a (Field_block i Value) from
               ∧ (s_field_used sol a (Field_block i Value) = true ->
                  rel field from field'))
          0 fields fields'
  | Value_boxed_number c =>
    ∃ (a : node) (bn : boxable_number) (from : node),
      may_come_from sol a n
      ∧ is_number_of_kind bn c = true
      ∧ g_constructor G a (Field_boxed_number bn) from
      ∧ v' = (if s_field_used sol a (Field_boxed_number bn) then v
              else Value_const (Poison Value))
  | Value_symbol s =>
    v' = v ∧ may_come_from sol (Node_symbol s) n
    ∧ (s_has_usage sol n = true -> s_has_usage sol (Node_symbol s) = true)
  | Value_immutable_array elements =>
    s_any_source sol n = true
    ∧ ∃ elements' : list value,
        v' = Value_immutable_array elements'
        ∧ values_related
            (fun w w' =>
               (w' = w
                ∧ ((∃ c : const, w = Value_const c)
                   ∨ ∃ s : symbol,
                       w = Value_symbol s
                       ∧ s_any_usage sol (Node_symbol s) = true))
               ∨ ∃ m : node, s_any_usage sol m = true ∧ rel w m w')
            elements elements'
  | Value_closure_of_set slot function_decls value_slots =>
    ∃ (sites : function_slot -> node)
      (value_slots' : list (value_slot * value)),
      may_come_from sol (sites slot) n
      ∧ v' = Value_closure_of_set slot function_decls value_slots'
      ∧ (exact -> value_slots' = value_slots)
      ∧ ∀ (f : function_slot) (id : code_id),
          lookup f function_decls = Some id ->
          may_come_from sol (sites f) (sites f)
          ∧ (∀ (g : function_slot) (id' : code_id),
               lookup g function_decls = Some id' ->
               g_constructor G (sites f) (Field_function_slot g) (sites g))
          ∧ slots_related
              (fun w x =>
                 kept w = true ->
                 ∃ from : node,
                   g_constructor G (sites f) (Field_value_slot w) from
                   ∧ (s_field_used sol (sites f) (Field_value_slot w) = true ->
                      ∃ x' : value,
                        lookup w value_slots' = Some x'
                        ∧ (s_has_usage sol from = true -> rel x from x')))
              value_slots []
  | Value_partial_application _ _ _ => v' = v
  | Value_closure _ _ | Value_location _ => False
  end.

(* A field of a block that may come from [a]. *)
Definition field_rel (a : node) (i : nat) (field field' : value) : Prop :=
  ∃ from : node,
    g_constructor G a (Field_block i Value) from
    ∧ (s_field_used sol a (Field_block i Value) = true ->
       rel field from field').

(* A value slot [w] of a closure that may come from [a], with the value
   slots [value_slots'] in the second run. *)
Definition slot_rel (a : node) (value_slots' : list (value_slot * value))
    (w : value_slot) (x : value) : Prop :=
  kept w = true ->
  ∃ from : node,
    g_constructor G a (Field_value_slot w) from
    ∧ (s_field_used sol a (Field_value_slot w) = true ->
       ∃ x' : value,
         lookup w value_slots' = Some x'
         ∧ (s_has_usage sol from = true -> rel x from x')).

Lemma rel_kind (v : value) (n : node) (v' : value) :
  rel v n v' -> kind_of_value v' = kind_of_value v.
Proof.
  destruct v; simpl; not_rel.
  - intros [_ ->]. reflexivity.
  - intros (sites & vs' & _ & -> & _). reflexivity.
  - intros [-> _]. reflexivity.
  - intros (a & fields' & _ & -> & _). reflexivity.
  - intros (_ & elements' & -> & _). reflexivity.
  - intros (a & bn & from & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - intros ->. reflexivity.
Qed.

Lemma rel_tagged (v : value) (n : node) (v' : value) :
  rel v n v' -> is_tagged_immediate v' = is_tagged_immediate v.
Proof.
  destruct v; simpl; not_rel.
  - intros [_ ->]. reflexivity.
  - intros (sites & vs' & _ & -> & _). reflexivity.
  - intros [-> _]. reflexivity.
  - intros (a & fields' & _ & -> & _). reflexivity.
  - intros (_ & elements' & -> & _). reflexivity.
  - intros (a & bn & from & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - intros ->. reflexivity.
Qed.

(* What the second run has is not a symbol unless the first has it. *)
Lemma rel_symbol (v : value) (n : node) (s : symbol) :
  rel v n (Value_symbol s) -> v = Value_symbol s.
Proof.
  destruct v; simpl; not_rel.
  - intros [_ Heq]. discriminate Heq.
  - intros (sites & vs' & _ & Heq & _). discriminate Heq.
  - intros [Heq _]. exact (eq_sym Heq).
  - intros (a & fields' & _ & Heq & _). discriminate Heq.
  - intros (_ & elements' & Heq & _). discriminate Heq.
  - intros (a & bn & from & _ & _ & _ & Heq).
    destruct (s_field_used sol a (Field_boxed_number bn)); discriminate Heq.
  - intros Heq. discriminate Heq.
Qed.

(* What [sim] relates is observed equal, and equal if [exact]. *)
Lemma sims_observed (ws : list value) :
  Forall (fun w => ∀ w' : value,
            sim w w' -> observed_equal w w' ∧ (exact -> w' = w)) ws ->
  ∀ ws' : list value,
    values_related (fun w w' => sim w w') ws ws' ->
    values_related observed_equal ws ws' ∧ (exact -> ws' = ws).
Proof.
  induction 1 as [ | w ws Hw _ IH]; intros [ | w' ws'] Hws; simpl in Hws |- *;
    try contradiction; [ split; [ exact I | reflexivity ] | ].
  destruct Hws as [Hw' Hrest].
  destruct (Hw w' Hw') as [Ho He]. destruct (IH ws' Hrest) as [Ho' He'].
  split; [ split; assumption | ].
  intros Hx. rewrite (He Hx), (He' Hx). reflexivity.
Qed.

Lemma sim_observed : ∀ v v' : value,
  sim v v' -> observed_equal v v' ∧ (exact -> v' = v).
Proof.
  apply (FlambdaHeapProofs.value_induction
           (fun v => ∀ v' : value,
              sim v v' -> observed_equal v v' ∧ (exact -> v' = v)));
    cbn [sim].
  - intros c v' ->. split; [ apply observed_equal_refl | reflexivity ].
  - intros code slots _ v' [].
  - intros slot decls vs _ v' (vs' & -> & Hex).
    split; [ cbn [observed_equal]; auto | ].
    intros He. rewrite (Hex He). reflexivity.
  - intros s v' [-> _]. split; [ apply observed_equal_refl | reflexivity ].
  - intros tag fields IH v' (fields' & -> & Hf).
    destruct (sims_observed fields IH fields' Hf) as [Ho He].
    split; [ cbn [observed_equal]; split; [ reflexivity | exact Ho ] | ].
    intros Hx. rewrite (He Hx). reflexivity.
  - intros elements IH v' (elements' & -> & Hf).
    destruct (sims_observed elements IH elements' Hf) as [Ho He].
    split; [ exact Ho | intros Hx; rewrite (He Hx); reflexivity ].
  - intros c v' ->. split; [ apply observed_equal_refl | reflexivity ].
  - intros l v' [].
  - intros callee args arity _ _ v' ->.
    split; [ apply observed_equal_refl | reflexivity ].
Qed.

Variable is_local : field -> bool.
Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.

Abbreviation as_ n := (s_any_source sol n = true).
Abbreviation au n := (s_any_usage sol n = true).
Abbreviation hs n := (s_has_source sol n = true).
Abbreviation hu n := (s_has_usage sol n = true).

Lemma block_field_not_local (i : nat) (kind : flambda_kind) :
  is_local (Field_block i kind) = false.
Proof. apply (not_local is_local Hlocal); intros; discriminate. Qed.

Lemma boxed_field_not_local (bn : boxable_number) :
  is_local (Field_boxed_number bn) = false.
Proof. apply (not_local is_local Hlocal); intros; discriminate. Qed.

(* Along an alias of the solution. *)
Lemma rel_alias (v : value) (to_ from : node) (v' : value) :
  s_alias sol to_ from = true -> rel v from v' -> rel v to_ v'.
Proof.
  intros Hal. destruct v; cbn [rel]; not_rel.
  - intros [Has ->]. split; [ | reflexivity ].
    exact (any_source_alias _ _ _ Hc _ _ Hal Has).
  - intros (sites & vs' & Ha & Hrest). exists sites, vs'.
    split; [ exact (may_come_from_alias _ _ _ Hc _ _ _ Hal Ha) | exact Hrest ].
  - intros (-> & Ha & Hhu). split; [ reflexivity | split ].
    + exact (may_come_from_alias _ _ _ Hc _ _ _ Hal Ha).
    + intros Hto. apply Hhu. exact (has_usage_alias _ _ _ Hc _ _ Hto Hal).
  - intros (a & fields' & Ha & Hrest). exists a, fields'.
    split; [ exact (may_come_from_alias _ _ _ Hc _ _ _ Hal Ha) | exact Hrest ].
  - intros [Has Hrest]. split; [ | exact Hrest ].
    exact (any_source_alias _ _ _ Hc _ _ Hal Has).
  - intros (a & bn & from' & Ha & Hrest). exists a, bn, from'.
    split; [ exact (may_come_from_alias _ _ _ Hc _ _ _ Hal Ha) | exact Hrest ].
  - exact (fun H => H).
Qed.

(* To a node with any source that has a usage only if [m] has any
   usage. *)
Lemma rel_escape (v : value) (to_ m : node) (v' : value) :
  as_ to_ -> (hu to_ -> au m) -> rel v m v' -> rel v to_ v'.
Proof.
  intros Has Hm. destruct v; cbn [rel]; not_rel.
  - intros [_ ->]. split; [ exact Has | reflexivity ].
  - intros (sites & vs' & Ha & Hrest). exists sites, vs'.
    split; [ exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm Ha)
           | exact Hrest ].
  - intros (-> & Ha & _). split; [ reflexivity | split ].
    + exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm Ha).
    + intros Hto. apply (au_hu _ _ _ Hc). destruct Ha as (_ & _ & _ & Hau & _).
      apply Hau, Hm, Hto.
  - intros (a & fields' & Ha & Hrest). exists a, fields'.
    split; [ exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm Ha)
           | exact Hrest ].
  - intros [_ Hrest]. split; [ exact Has | exact Hrest ].
  - intros (a & bn & from' & Ha & Hrest). exists a, bn, from'.
    split; [ exact (may_come_from_escape _ _ _ Hc _ _ _ Has Hm Ha)
           | exact Hrest ].
  - exact (fun H => H).
Qed.

(* The read of the field [f] of a value that may come from [a], as
   FlambdaReaperFlowProofs.flows_read. *)
Lemma rel_read (a base to_ from : node) (f : field) (v v' : value) :
  is_local f = false ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  rel v from v' -> rel v to_ v'.
Proof.
  intros Hl Hcons Hacc (_ & Hsrc & _ & Hau & Hus) Hv.
  destruct (s_any_source sol base) eqn:Eb.
  - pose proof (any_source_accessor _ _ _ Hc _ _ _ Eb Hacc Hl) as Has.
    destruct (s_any_usage sol a) eqn:Ea.
    + apply (rel_escape v to_ from v' Has); [ | exact Hv ].
      intros _. exact (any_usage_constructor _ _ _ Hc _ _ _ Ea Hcons Hl).
    + destruct (s_has_usage sol to_) eqn:Ehu.
      * pose proof (has_usage_accessor _ _ _ Hc _ _ _ Ehu Hacc) as Hhu.
        assert (Eau : s_any_usage sol base = false).
        { destruct (s_any_usage sol base);
          [ discriminate (Hau eq_refl) | reflexivity ]. }
        pose proof (usages_accessor _ _ _ Hc _ _ _ Hacc Eau) as Hbb.
        destruct (Hus Hhu base Hbb) as [Hab | Hab]; [ | congruence ].
        apply (rel_alias v to_ from v'); [ | exact Hv ].
        exact (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hab Ea Hacc).
      * apply (rel_escape v to_ from v' Has); [ | exact Hv ].
        intros Hhu. congruence.
  - destruct Hsrc as [Hsrc | Hsrc]; [ discriminate | ].
    apply (rel_alias v to_ from v'); [ | exact Hv ].
    exact (alias_accessor_source _ _ _ Hc _ _ _ _ _ Hacc Hsrc Eb Hcons).
Qed.

(* At a node with any usage, what the second run has is [sim]. *)
Lemma rel_au_sim : ∀ (v : value) (n : node) (v' : value),
  au n -> rel v n v' -> sim v v'.
Proof.
  apply (FlambdaHeapProofs.value_induction
           (fun v => ∀ (n : node) (v' : value),
              au n -> rel v n v' -> sim v v'));
    cbn [rel sim].
  - intros c n v' _ [_ ->]. reflexivity.
  - intros code slots _ n v' _ [].
  - intros slot decls vs _ n v' _ (sites & vs' & _ & -> & Hex & _).
    exists vs'. split; [ reflexivity | exact Hex ].
  - intros s n v' Hn (-> & Ha & _). split; [ reflexivity | ].
    destruct Ha as (_ & _ & _ & Hau & _). exact (Hau Hn).
  - intros tag fields IH n v' Hn (a & fields' & Ha & -> & Hfields).
    exists fields'. split; [ reflexivity | ].
    destruct Ha as (_ & _ & _ & Hau & _). specialize (Hau Hn).
    revert fields' Hfields. generalize 0 as k.
    induction IH as [ | field fields Hfield _ IHf]; intros k [ | field' fields']
      Hfields; simpl in Hfields |- *; try contradiction; [ exact I | ].
    destruct Hfields as [(from & Hcons & Hused) Hrest]. split.
    + pose proof (block_field_not_local k Value) as Hl.
      apply (Hfield from);
        [ exact (any_usage_constructor _ _ _ Hc _ _ _ Hau Hcons Hl) | ].
      apply Hused. exact (field_used_any_usage _ _ _ Hc _ _ _ Hcons Hau Hl).
    + exact (IHf (S k) fields' Hrest).
  - intros elements IH n v' _ (_ & elements' & -> & Helements).
    exists elements'. split; [ reflexivity | ].
    revert elements' Helements.
    induction IH as [ | w elements Hw _ IHe]; intros [ | w' elements']
      Helements; simpl in Helements |- *; try contradiction; [ exact I | ].
    destruct Helements as [Hel Hrest]. split; [ | exact (IHe elements' Hrest) ].
    destruct Hel as [(-> & [(c & ->) | (s & -> & Hs)]) | (m & Hm & Hrel)].
    + reflexivity.
    + split; [ reflexivity | exact Hs ].
    + exact (Hw m w' Hm Hrel).
  - intros c n v' Hn (a & bn & from & (_ & _ & _ & Hau & _) & _ & Hcons & ->).
    rewrite (field_used_any_usage _ _ _ Hc _ _ _ Hcons (Hau Hn)
               (boxed_field_not_local bn)).
    reflexivity.
  - intros l n v' _ [].
  - intros callee args arity _ _ n v' _ ->. reflexivity.
Qed.

Hypothesis Hexact : field_used_exact is_local G sol.

(* A field that is used holds what has a use, whichever of its
   constructor facts. *)
Lemma field_used_hu (a from : node) (f : field) :
  is_local f = false ->
  s_field_used sol a f = true -> g_constructor G a f from -> hu from.
Proof.
  intros Hl Hused Hcons.
  destruct (s_any_usage sol a) eqn:Ea.
  - exact (au_hu _ _ _ Hc _ (any_usage_constructor _ _ _ Hc _ _ _ Ea Hcons Hl)).
  - destruct (Hexact a f Hused)
      as (from0 & _ & [[Hau _] | [(usage & v & Hus & Hacc & Hhu)
                                | [(x & Hau & _) | (_ & _ & _ & Hl' & _)]]]);
      [ congruence | | congruence | congruence ].
    apply (has_usage_alias _ _ _ Hc v from Hhu).
    exact (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hus Ea Hacc).
Qed.

End Rel.

(* ======================================================================== *)
(* What the unit asks of an expression                                      *)
(* ======================================================================== *)

(* The [Switch]es of the expression, and of its handlers, are on what has
   any usage, as the traversal makes them ([cond_any_usage]). *)
Fixpoint scrutinees_used (G : graph) (e : expr) : Prop :=
  match e with
  | Let _ _ _ body | Let_static _ body => scrutinees_used G body
  | Let_cont_non_recursive _ h body =>
    handler_scrutinees_used G h ∧ scrutinees_used G body
  | Let_cont_recursive handlers body =>
    every (fun '(_, h) => handler_scrutinees_used G h) handlers
    ∧ scrutinees_used G body
  | Apply _ | Apply_cont _ | Invalid => True
  | Switch scrutinee _ => g_any_usage G (node_of_simple scrutinee)
  end
with handler_scrutinees_used (G : graph) (h : continuation_handler)
    : Prop :=
  match h with
  | Handler _ _ handler => scrutinees_used G handler
  end.

(* A [Let] that is not of a set of closures. *)
Definition named_plain (d : named) : bool :=
  match d with
  | Set_of_closures _ _ | Closure_of_set _ _ _ => false
  | _ => true
  end.

(* The check of the units of the proof: [no_code_no_slots] if the
   rebuild is compared exactly ([exact]), [no_call] otherwise. *)
Definition named_checked (exact : bool) (kind : flambda_kind) (d : named)
    : bool :=
  if exact then named_no_code_no_slots d else named_no_code kind d.

Definition binding_checked (exact : bool) (binding : static_binding)
    : bool :=
  if exact then binding_no_code_no_slots binding else binding_no_call binding.

Fixpoint checked (exact : bool) (e : expr) : bool :=
  match e with
  | Let _ kind d body => named_checked exact kind d && checked exact body
  | Let_static bindings body =>
    forallb (binding_checked exact) bindings && checked exact body
  | Let_cont_non_recursive _ h body =>
    handler_checked exact h && checked exact body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_checked exact h) handlers
    && checked exact body
  | Apply _ => false
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_checked (exact : bool) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => checked exact handler
  end.

Lemma named_checked_no_code (exact : bool) (kind : flambda_kind) (d : named) :
  named_checked exact kind d = true -> named_no_code kind d = true.
Proof.
  intros H. destruct exact; [ | exact H ].
  destruct d; simpl in H |- *; try discriminate H; reflexivity.
Qed.

(* A binding of the check: of no code, or of code whose body is of no
   code, up to observation. *)
Lemma binding_checked_cases (exact : bool) (b : static_binding) :
  binding_checked exact b = true ->
  binding_no_code b = true
  ∨ ∃ (id : code_id) (c : code),
      b = Bind_code id c ∧ exact = false ∧ no_code (code_body c) = true.
Proof.
  intros H. destruct exact.
  - left. destruct b as [ | | s [ | slot syms decls [ | ] | | | | ]];
      simpl in H |- *; try discriminate H; try reflexivity; exact H.
  - destruct b as [id c | id | s const]; simpl in H.
    + right. exists id, c. auto.
    + left. reflexivity.
    + left. exact H.
Qed.

(* A unit without code meets [no_call]. *)
Lemma no_code_no_call_all :
  (∀ e : expr, no_code e = true -> no_call e = true)
  ∧ (∀ h : continuation_handler,
       handler_no_code h = true -> handler_no_call h = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hd Hb]. rewrite Hd, (IH Hb). reflexivity.
  - intros bindings body _ IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin.
    specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; simpl in Hbs |- *;
      [ discriminate Hbs | reflexivity | exact Hbs ].
  - intros k h body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. exact H.
  - intros ac H. exact H.
  - intros scrutinee arms H. exact H.
  - intros H. exact H.
  - intros params metadata handler IH H. exact (IH H).
Qed.

(* A body of the check of no code is of the check, up to observation. *)
Lemma no_code_checked (e : expr) :
  no_code e = true -> checked false e = true.
Proof.
  enough (H : (∀ e : expr, checked false e = no_call e)
              ∧ (∀ h : continuation_handler,
                   handler_checked false h = handler_no_call h))
    by (intros He; rewrite (proj1 H); exact (proj1 no_code_no_call_all e He)).
  apply expr_handler_ind; simpl.
  - intros x kind d body IH. rewrite IH. reflexivity.
  - intros bindings body _ IH. rewrite IH. reflexivity.
  - intros k h body IHh IH. rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH. rewrite IH. f_equal.
    induction handlers as [ | [k h] handlers IHl]; [ reflexivity | ].
    simpl. rewrite (IHh k h (or_introl eq_refl)).
    rewrite IHl; [ reflexivity | ].
    intros k' h' Hin. exact (IHh k' h' (or_intror Hin)).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH. exact IH.
Qed.

(* A primitive whose value may be of kind [Region] or [Rec_info] is the
   identity. *)
Lemma region_prim (p : prim) :
  prim_result_kind p = Region ∨ prim_result_kind p = Rec_info ->
  ∃ k : flambda_kind, p = Unary (Opaque_identity k).
Proof.
  intros H.
  destruct p as [up | bp | vp | hp];
    [ destruct up | destruct bp | destruct vp | destruct hp ];
    cbn [prim_result_kind result_kind_of_unary_primitive
         result_kind_of_binary_primitive result_kind_of_variadic_primitive
         result_kind_of_heap_primitive] in H;
    try (eexists; reflexivity);
    unfold naked_immediate_kind, kind_of_standard_int, unboxed_kind in H;
    repeat (match type of H with
            | context [match ?x with _ => _ end] => destruct x
            end);
    destruct H as [H | H]; discriminate H.
Qed.

(* A value of another kind than [Value] is a constant. *)
Lemma not_value_const (v : value) :
  kind_of_value v <> Value -> ∃ c : const, v = Value_const c.
Proof.
  destruct v; simpl; intros H; try (destruct (H eq_refl)).
  eexists. reflexivity.
Qed.

Lemma region_not_value (k : flambda_kind) :
  k = Region ∨ k = Rec_info -> k <> Value.
Proof. intros [-> | ->]; discriminate. Qed.

(* The [Let]s and the non-recursive [Let_cont]s that begin an expression:
   what the rebuild may remove without a step. *)
Fixpoint spine (e : expr) : nat :=
  match e with
  | Let _ _ _ body | Let_cont_non_recursive _ _ body => S (spine body)
  | _ => 0
  end.

Definition measure (s : state) : nat := spine (state_expr s).

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Section Run.

(* The setting of FlambdaReaperFlowProofs.flow_invariant, and the
   arguments of the rebuild. *)
Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable keep : value_slot -> bool.
Variable codes : list (code_id * code).
Variable symbols : list symbol.
Variable u : unit_interface.
Variable F : facts.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable first_fresh : positive.
Variable exact : bool.

Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hexact : field_used_exact is_local G sol.
Hypothesis Hconsts : g_any_source G Node_all_constants.
Hypothesis Hsyms :
  ∀ s : symbol, noted s -> may_come_from sol (Node_symbol s) (Node_symbol s).
Hypothesis Hunit_syms :
  ∀ s : symbol,
    among s symbols = true -> may_come_from sol (Node_symbol s) (Node_symbol s).
Hypothesis Hdeps : ∀ d : closure_dep, deps d -> closure_dep_facts G cds d.
Hypothesis Hnd : NoDup (map fst cds).
Hypothesis Hcds :
  ∀ (id : code_id) (cd : code_dep),
    lookup id cds = Some cd -> known_witness_edges G id cd.
Hypothesis HG : G = graph_of_facts F.
Hypothesis Hunit :
  ∀ (k : continuation) (p : node), p ∈ unit_cont_nodes u k -> g_any_usage G p.

Abbreviation slots := (∀ f : field, is_local f = false).
Abbreviation called := (called_by G cds).
Abbreviation flows_ := (flows G sol keep called).
Abbreviation env_flows_ := (env_flows G sol keep called).
Abbreviation statics_flow_ := (statics_flow G sol keep called).
Abbreviation edges :=
  (expr_edges G noted deps cds keep codes symbols slots None).
Abbreviation handler_edges_ :=
  (handler_edges G noted deps cds keep codes symbols slots None).
Abbreviation binding_edges_ :=
  (binding_edges G noted deps cds keep codes symbols slots).
Abbreviation named_edges_ := (named_edges G noted deps keep slots).
Abbreviation as_ n := (s_any_source sol n = true).
Abbreviation au n := (s_any_usage sol n = true).
Abbreviation hs n := (s_has_source sol n = true).
Abbreviation hu n := (s_has_usage sol n = true).
Abbreviation rel := (rel G sol keep (exact = true)).
Abbreviation field_rel := (field_rel G sol keep (exact = true)).
Abbreviation slot_rel := (slot_rel G sol keep (exact = true)).
Abbreviation sim := (sim sol (exact = true)).
Abbreviation rebuild :=
  (rebuild_expr F sol preserve_direct_calls unit_code_ids first_fresh).
Abbreviation rebuild_h :=
  (rebuild_handler F sol preserve_direct_calls unit_code_ids first_fresh).
Abbreviation rebuild_b :=
  (rebuild_binding F sol preserve_direct_calls unit_code_ids first_fresh).
Abbreviation rebuild_c :=
  (rebuild_code F sol preserve_direct_calls unit_code_ids first_fresh).

(* ------------------------------------------------------------------------ *)
(* Environments                                                             *)
(* ------------------------------------------------------------------------ *)

(* Every variable of the first run has its kind in [kinds], and, if the
   rebuild keeps it, is bound in the second to a related value. *)
Definition env_ok (kinds : finite_map flambda_kind) (vars vars' : env)
    : Prop :=
  ∀ (x : variable) (v : value),
    lookup x vars = Some v ->
    lookup x kinds = Some (kind_of_value v)
    ∧ (is_var_used sol x (kind_of_value v) = true ->
       ∃ v' : value, lookup x vars' = Some v' ∧ rel v (Node_variable x) v').

(* What the second static environment binds, the first does; a symbol
   with a use is bound in both to related blocks; and the first binds
   only blocks and closures of static sets, which the second binds with
   related value slots or not at all, and binds if they have a use; the
   second binds the code of the first, rebuilt, or none. *)
Definition statics_rel (statics statics' : static_env) : Prop :=
  (∀ s : symbol,
     lookup s (static_values statics) = None ->
     lookup s (static_values statics') = None)
  ∧ (∀ (s : symbol) (tag : Z) (fields : list value),
       lookup s (static_values statics) = Some (Static_block tag fields) ->
       hu (Node_symbol s) ->
       ∃ fields' : list value,
         lookup s (static_values statics') = Some (Static_block tag fields')
         ∧ fields_related (field_rel (Node_symbol s)) 0 fields fields')
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
       lookup s (static_values statics) = Some allocated ->
       (∃ (tag : Z) (fields : list value),
          allocated = Static_block tag fields)
       ∨ ∃ (slot : function_slot)
           (closure_symbols : list (function_slot * symbol))
           (function_decls : list (function_slot * code_id))
           (vs : list (value_slot * value)),
           allocated
           = Static_closure_of_set slot closure_symbols function_decls vs
           ∧ (lookup s (static_values statics') = None
              ∨ ∃ vs' : list (value_slot * value),
                  lookup s (static_values statics')
                  = Some (Static_closure_of_set slot closure_symbols
                            function_decls vs')
                  ∧ (exact = true -> vs' = vs)
                  ∧ slots_related (slot_rel (Node_symbol s) vs') vs [])
           ∧ (hu (Node_symbol s) ->
              lookup s (static_values statics') <> None))
  ∧ (∀ id : code_id,
       lookup id (static_code statics') = None
       ∨ lookup id (static_code statics')
         = option_map rebuild_c (lookup id (static_code statics))).

(* The simples that [rewrite_simple] keeps. *)
Definition kept_simple (en : rebuild_env) (s : simple) : bool :=
  match s with
  | Var x => is_var_used sol x (kind_of_variable en x)
  | Symbol sym => is_symbol_used sol sym
  | Const _ => true
  end.

Lemma is_var_used_hu (x : variable) (k : flambda_kind) :
  hu (Node_variable x) -> is_var_used sol x k = true.
Proof. intros H. destruct k; simpl; auto. Qed.

Lemma is_var_used_cases (x : variable) (k : flambda_kind) :
  is_var_used sol x k = true ->
  hu (Node_variable x) ∨ (k = Region ∨ k = Rec_info).
Proof. destruct k; simpl; auto. Qed.

Lemma kept_of_hu (en : rebuild_env) (s : simple) :
  hu (node_of_simple s) -> kept_simple en s = true.
Proof.
  destruct s as [x | sym | c]; simpl; intros H; [ | exact H | reflexivity ].
  apply is_var_used_hu, H.
Qed.

Lemma env_ok_kind (en : rebuild_env) (vars vars' : env) (x : variable)
    (v : value) :
  env_ok (env_kinds en) vars vars' -> lookup x vars = Some v ->
  kind_of_variable en x = kind_of_value v.
Proof.
  intros Henv Hx. unfold kind_of_variable.
  rewrite (proj1 (Henv x v Hx)). reflexivity.
Qed.

(* A simple whose value is of kind [Region] or [Rec_info] is kept. *)
Lemma kept_of_region (en : rebuild_env) (vars vars' : env) (s : simple)
    (v : value) :
  env_ok (env_kinds en) vars vars' -> eval_simple vars s = Some v ->
  kind_of_value v = Region ∨ kind_of_value v = Rec_info ->
  kept_simple en s = true.
Proof.
  intros Henv Hs Hk. destruct s as [x | sym | c]; simpl in Hs |- *;
    [ | | reflexivity ].
  - rewrite (env_ok_kind en vars vars' x v Henv Hs).
    destruct Hk as [-> | ->]; reflexivity.
  - injection Hs as <-. simpl in Hk. destruct Hk; discriminate.
Qed.

(* A simple of the first run, and what the rebuild makes of it in the
   second: of the same kind, and related if it is kept. *)
Lemma rewrite_simple_eval (en : rebuild_env) (vars vars' : env) (s : simple)
    (v : value) :
  env_ok (env_kinds en) vars vars' ->
  simple_noted noted s ->
  eval_simple vars s = Some v ->
  ∃ v' : value,
    eval_simple vars' (rewrite_simple sol en s) = Some v'
    ∧ kind_of_value v' = kind_of_value v
    ∧ (kept_simple en s = true ->
       rewrite_simple sol en s = s ∧ rel v (node_of_simple s) v').
Proof.
  intros Henv Hnoted Hs.
  destruct s as [x | sym | c]; simpl in Hs, Hnoted.
  - destruct (Henv x v Hs) as [_ Hused].
    pose proof (env_ok_kind en vars vars' x v Henv Hs) as Hk.
    unfold rewrite_simple, kept_simple. cbv zeta. rewrite Hk.
    destruct (is_var_used sol x (kind_of_value v)) eqn:Hu.
    + destruct (Hused eq_refl) as (v' & Hl & Hrel). exists v'.
      split; [ exact Hl | split; [ exact (rel_kind _ _ _ _ _ _ _ Hrel) | ] ].
      intros _. split; [ reflexivity | exact Hrel ].
    + exists (Value_const (Poison (kind_of_value v))).
      split; [ reflexivity | split; [ reflexivity | discriminate ] ].
  - injection Hs as <-. unfold rewrite_simple, kept_simple.
    destruct (is_symbol_used sol sym) eqn:Hu.
    + exists (Value_symbol sym).
      split; [ reflexivity | split; [ reflexivity | ] ].
      intros _. split; [ reflexivity | ].
      split; [ reflexivity | split; [ exact (Hsyms sym Hnoted) | auto ] ].
    + exists (Value_const (Poison Value)).
      split; [ reflexivity | split; [ reflexivity | discriminate ] ].
  - injection Hs as <-. exists (Value_const c).
    split; [ reflexivity | split; [ reflexivity | ] ].
    intros _. split; [ reflexivity | ].
    split; [ exact (consts_any_source is_local G sol Hc Hconsts)
           | reflexivity ].
Qed.

Lemma rewrite_simples_eval (en : rebuild_env) (vars vars' : env) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (args : list simple) (vs : list value),
    simples_noted noted args ->
    eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' (map (rewrite_simple sol en) args) = Some vs'
      ∧ map kind_of_value vs' = map kind_of_value vs
      ∧ ∀ (i : nat) (s : simple) (v v' : value),
          nth_error args i = Some s -> nth_error vs i = Some v ->
          nth_error vs' i = Some v' -> kept_simple en s = true ->
          rel v (node_of_simple s) v'.
Proof.
  intros Henv. induction args as [ | s args IH]; intros vs Hnoted Hvs;
    simpl in Hvs.
  - injection Hvs as <-. exists [].
    split; [ reflexivity | split; [ reflexivity | ] ].
    intros [ | i]; discriminate.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
    injection Hvs as <-.
    destruct (rewrite_simple_eval en vars vars' s v Henv
                (Hnoted s (or_introl eq_refl)) Hs)
      as (v' & Hs' & Hk & Hkept).
    destruct (IH ws (fun s' Hin => Hnoted s' (or_intror Hin)) eq_refl)
      as (ws' & Hws' & Hks & Hrels).
    exists (v' :: ws'). simpl. rewrite Hs', Hws'.
    split; [ reflexivity | split; [ simpl; rewrite Hk, Hks; reflexivity | ] ].
    intros [ | i] s'' w w' Hi Hw Hw'; simpl in Hi, Hw, Hw'.
    + injection Hi as <-. injection Hw as <-. injection Hw' as <-.
      intros Hkeep. exact (proj2 (Hkept Hkeep)).
    + exact (Hrels i s'' w w' Hi Hw Hw').
Qed.

(* A binding of the first run, and maybe of the second. *)
Lemma env_ok_bind (kinds : finite_map flambda_kind) (vars vars' vars'' : env)
    (x : variable) (v : value) :
  env_ok kinds vars vars' ->
  (is_var_used sol x (kind_of_value v) = true ->
   ∃ v' : value, vars'' = (x, v') :: vars' ∧ rel v (Node_variable x) v') ->
  (vars'' = vars' ∨ ∃ w : value, vars'' = (x, w) :: vars') ->
  env_ok ((x, kind_of_value v) :: kinds) ((x, v) :: vars) vars''.
Proof.
  intros Henv Hx Hvars'' y w Hy. simpl in Hy |- *.
  destruct (Pos.eqb_spec y x) as [-> | Hne].
  - injection Hy as <-. split; [ reflexivity | ].
    intros Hused. destruct (Hx Hused) as (v' & -> & Hrel).
    exists v'. simpl. rewrite Pos.eqb_refl. auto.
  - destruct (Henv y w Hy) as [Hk Hused]. split; [ exact Hk | ].
    intros Hu. destruct (Hused Hu) as (v' & Hl & Hrel). exists v'.
    split; [ | exact Hrel ].
    destruct Hvars'' as [-> | (w' & ->)]; [ exact Hl | ].
    simpl. rewrite (proj2 (Pos.eqb_neq y x) Hne). exact Hl.
Qed.

(* ------------------------------------------------------------------------ *)
(* Static environments                                                      *)
(* ------------------------------------------------------------------------ *)

(* A block read off a value, in the two runs. *)
Lemma block_of_rel (statics statics' : static_env) (v v' : value) (n : node)
    (tag : Z) (fields : list value) :
  statics_rel statics statics' -> rel v n v' -> hu n ->
  block_of statics v = Some (tag, fields) ->
  ∃ (a : node) (fields' : list value),
    block_of statics' v' = Some (tag, fields')
    ∧ may_come_from sol a n
    ∧ fields_related (field_rel a) 0 fields fields'.
Proof.
  intros [_ [Hblocks _]] Hrel Hn Hb.
  destruct (block_of_cases _ _ _ _ Hb) as [-> | (s & -> & Hl)].
  - destruct Hrel as (a & fields' & Ha & -> & Hf).
    exists a, fields'. auto.
  - destruct Hrel as (-> & Ha & Hhu).
    destruct (Hblocks s tag fields Hl (Hhu Hn)) as (fields' & Hl' & Hf).
    exists (Node_symbol s), fields'.
    split; [ simpl; rewrite Hl'; reflexivity | auto ].
Qed.

Lemma block_of_none_rel (statics statics' : static_env) (v v' : value)
    (n : node) :
  statics_rel statics statics' ->
  rel v n v' -> block_of statics v = None -> block_of statics' v' = None.
Proof.
  intros [Hnone [_ [Honly _]]] Hrel Hb.
  destruct v as [c | | | s | tag fields | elements | c | | ];
    cbn [rel] in Hrel; try contradiction.
  - destruct Hrel as [_ ->]. exact Hb.
  - destruct Hrel as (sites & vs' & _ & -> & _). reflexivity.
  - destruct Hrel as [-> _]. simpl in Hb |- *.
    destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl.
    + destruct (Honly s allocated Hl)
        as [(tag & fields & ->)
           | (slot & syms & decls & vs & -> & [Hn | (vs' & Hs & _)] & _)];
        [ discriminate Hb | rewrite Hn; reflexivity | rewrite Hs; exact Hb ].
    + rewrite (Hnone s Hl). reflexivity.
  - discriminate Hb.
  - destruct Hrel as (_ & elements' & -> & _). reflexivity.
  - destruct Hrel as (a & bn & from & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - rewrite Hrel. exact Hb.
Qed.

(* A symbol among the arguments of a primitive in the second run is
   bound if it is in the first. *)
Lemma bound_if_symbol_rel (statics statics' : static_env) (v v' : value)
    (n : node) :
  statics_rel statics statics' ->
  rel v n v' -> hu n ->
  bound_if_symbol statics v = true -> bound_if_symbol statics' v' = true.
Proof.
  intros [_ [Hblocks [Honly _]]] Hrel Hn Hb.
  destruct v' as [ | | | s | | | | | ]; try reflexivity.
  pose proof (rel_symbol _ _ _ _ _ _ _ Hrel) as ->.
  destruct Hrel as (_ & _ & Hhu). simpl in Hb |- *.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl;
    [ | discriminate Hb ].
  destruct (Honly s allocated Hl)
    as [(tag & fields & ->) | (slot & syms & decls & vs & -> & _ & Hs)].
  - destruct (Hblocks s tag fields Hl (Hhu Hn)) as (fields' & Hl' & _).
    rewrite Hl'. reflexivity.
  - destruct (lookup s (static_values statics')); [ reflexivity | ].
    exfalso. exact (Hs (Hhu Hn) eq_refl).
Qed.

(* The fields of a block whose allocation has any usage are [sim]. *)
Lemma fields_sim (a : node) (k : nat) (fields fields' : list value) :
  au a -> fields_related (field_rel a) k fields fields' ->
  values_related sim fields fields'.
Proof.
  intros Hau. revert k fields'.
  induction fields as [ | field fields IH]; intros k [ | field' fields']
    Hfields; simpl in Hfields |- *; try contradiction; [ exact I | ].
  destruct Hfields as [(from & Hcons & Hused) Hrest]. split.
  - pose proof (block_field_not_local is_local Hlocal k Value) as Hl.
    apply (rel_au_sim G sol keep (exact = true) is_local Hc Hlocal field from
             field');
      [ exact (any_usage_constructor _ _ _ Hc _ _ _ Hau Hcons Hl) | ].
    apply Hused. exact (field_used_any_usage _ _ _ Hc _ _ _ Hcons Hau Hl).
  - exact (IH _ _ Hrest).
Qed.

(* What a symbol with any usage is bound to in the two runs. *)
Lemma statics_au (statics statics' : static_env) (s : symbol) :
  statics_rel statics statics' ->
  au (Node_symbol s) ->
  (lookup s (static_values statics) = None ->
   lookup s (static_values statics') = None)
  ∧ (∀ (tag : Z) (fields : list value),
       lookup s (static_values statics) = Some (Static_block tag fields) ->
       ∃ fields' : list value,
         lookup s (static_values statics') = Some (Static_block tag fields')
         ∧ values_related sim fields fields')
  ∧ (∀ (slot : function_slot) (syms : list (function_slot * symbol))
       (decls : list (function_slot * code_id))
       (vs : list (value_slot * value)),
       lookup s (static_values statics)
       = Some (Static_closure_of_set slot syms decls vs) ->
       ∃ vs' : list (value_slot * value),
         lookup s (static_values statics')
         = Some (Static_closure_of_set slot syms decls vs'))
  ∧ (∀ allocated : statically_allocated_value,
       lookup s (static_values statics) = Some allocated ->
       (∃ (tag : Z) (fields : list value),
          allocated = Static_block tag fields)
       ∨ ∃ (slot : function_slot) (syms : list (function_slot * symbol))
           (decls : list (function_slot * code_id))
           (vs : list (value_slot * value)),
           allocated = Static_closure_of_set slot syms decls vs).
Proof.
  intros [Hnone [Hblocks [Honly _]]] Hau.
  pose proof (au_hu _ _ _ Hc _ Hau) as Hhu.
  split; [ exact (Hnone s) | split; [ | split ] ].
  - intros tag fields Hl.
    destruct (Hblocks s tag fields Hl Hhu) as (fields' & Hl' & Hf).
    exists fields'. split; [ exact Hl' | exact (fields_sim _ _ _ _ Hau Hf) ].
  - intros slot syms decls vs Hl.
    destruct (Honly s _ Hl)
      as [(tag & fields & Heq)
         | (slot' & syms' & decls' & vs0 & Heq & Hs & Hb)];
      [ discriminate Heq | ].
    injection Heq as <- <- <- <-.
    destruct Hs as [Hn | (vs' & Hs & _)]; [ | exists vs'; exact Hs ].
    exfalso. exact (Hb Hhu Hn).
  - intros allocated Hl.
    destruct (Honly s _ Hl)
      as [Hb | (slot & syms & decls & vs & Heq & _)]; [ left; exact Hb | ].
    right. exists slot, syms, decls, vs. exact Heq.
Qed.

(* ------------------------------------------------------------------------ *)
(* The fields of a block                                                    *)
(* ------------------------------------------------------------------------ *)

(* The fields of a block made at [base], from field [k]: a field used is
   related; one not used is poison. *)
Lemma rebuild_fields_eval (en : rebuild_env) (vars vars' : env)
    (base : node) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (fields : list simple) (k : nat) (vs : list value),
    (∀ (i : nat) (s : simple),
       nth_error fields i = Some s ->
       g_constructor G base (Field_block (k + i) Value) (node_of_simple s)) ->
    simples_noted noted fields ->
    eval_simples vars fields = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' (rebuild_fields sol en base k fields) = Some vs'
      ∧ fields_related (field_rel base) k vs vs'
      ∧ ∀ (i : nat) (v v' : value),
          nth_error vs i = Some v -> nth_error vs' i = Some v' ->
          v' = Value_const (Poison Value)
          ∨ ∃ n : node, rel v n v' ∧ hu n.
Proof.
  intros Henv. induction fields as [ | s fields IH]; intros k vs Hcons Hnoted
    Hvs; simpl in Hvs.
  - injection Hvs as <-. exists [].
    split; [ reflexivity | split; [ exact I | ] ].
    intros [ | i]; discriminate.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars fields) as [ws | ] eqn:Hws;
      [ | discriminate ].
    injection Hvs as <-.
    assert (Hs_cons : g_constructor G base (Field_block k Value)
                        (node_of_simple s)).
    { rewrite <- (Nat.add_0_r k). exact (Hcons 0 s eq_refl). }
    assert (Hcons' : ∀ (i : nat) (s' : simple),
              nth_error fields i = Some s' ->
              g_constructor G base (Field_block (S k + i) Value)
                (node_of_simple s')).
    { intros i s' Hi. replace (S k + i) with (k + S i) by lia.
      exact (Hcons (S i) s' Hi). }
    destruct (IH (S k) ws Hcons' (fun s' Hin => Hnoted s' (or_intror Hin))
                eq_refl)
      as (ws' & Hws' & Hf & Hpoint).
    destruct (s_field_used sol base (Field_block k Value)) eqn:Hused.
    + pose proof (field_used_hu G sol is_local Hc Hexact base
                    (node_of_simple s) _
                    (block_field_not_local is_local Hlocal k Value)
                    Hused Hs_cons) as Hhu.
      destruct (rewrite_simple_eval en vars vars' s v Henv
                  (Hnoted s (or_introl eq_refl)) Hs)
        as (v' & Hs' & _ & Hkept).
      destruct (Hkept (kept_of_hu en s Hhu)) as [_ Hrel].
      exists (v' :: ws'). cbn [rebuild_fields eval_simples].
      unfold field_used. rewrite Hused, Hs', Hws'.
      split; [ reflexivity | split ].
      * split; [ | exact Hf ].
        exists (node_of_simple s). split; [ exact Hs_cons | auto ].
      * intros [ | i] w w' Hw Hw'; simpl in Hw, Hw';
          [ | exact (Hpoint i w w' Hw Hw') ].
        injection Hw as <-. injection Hw' as <-. right.
        exists (node_of_simple s). auto.
    + exists (Value_const (Poison Value) :: ws').
      cbn [rebuild_fields eval_simples]. unfold field_used.
      rewrite Hused, Hws'.
      split; [ reflexivity | split ].
      * split; [ | exact Hf ].
        exists (node_of_simple s).
        split; [ exact Hs_cons | intros Hu; congruence ].
      * intros [ | i] w w' Hw Hw'; simpl in Hw, Hw';
          [ | exact (Hpoint i w w' Hw Hw') ].
        injection Hw' as <-. left. reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Primitives                                                               *)
(* ------------------------------------------------------------------------ *)

(* A unary primitive on a simple that the rebuild keeps. *)
Lemma unary_target (en : rebuild_env) (vars' : env) (statics' : static_env)
    (up : unary_primitive) (s1 : simple) (v1' : value) :
  rewrite_simple sol en s1 = s1 -> eval_simple vars' s1 = Some v1' ->
  eval_named statics' vars' (Prim (Unary up) (map (rewrite_simple sol en) [s1]))
  = eval_unary_primitive statics' up v1'.
Proof.
  intros Hr Hs. cbn [map eval_named eval_simples]. rewrite Hr, Hs.
  cbv beta iota. apply eval_prim_unary.
Qed.

(* A load of a field of a block, into a variable with a use. *)
Lemma block_load_rel (statics statics' : static_env) (x : variable)
    (base : node) (kind : block_access_kind) (fld : Z) (v1 v1' v : value) :
  statics_rel statics statics' ->
  g_accessor G (Node_variable x) (Field_block (Z.to_nat fld) Value) base ->
  rel v1 base v1' -> hu (Node_variable x) ->
  eval_block_load statics kind fld v1 = Some v ->
  ∃ v' : value,
    eval_block_load statics' kind fld v1' = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Hst Hacc Hrel Hx Hev.
  pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc) as Hbase.
  destruct kind as [tag size fk]. unfold eval_block_load in Hev |- *.
  destruct (block_of statics v1) as [[btag fields] | ] eqn:Hb;
    [ | discriminate Hev ].
  destruct (block_of_rel statics statics' v1 v1' base btag fields Hst Hrel
              Hbase Hb)
    as (a & fields' & Hb' & Ha & Hf).
  rewrite Hb'. cbv beta iota in Hev |- *.
  rewrite (fields_related_length _ _ _ _ Hf).
  destruct (known_is tag btag && known_is size (Z.of_nat (length fields))
            && (0 <=? fld)%Z);
    [ | discriminate Hev ].
  destruct (nth_error fields (Z.to_nat fld)) as [loaded | ] eqn:Hnth;
    [ | discriminate Hev ].
  destruct (fields_related_nth _ _ _ _ _ _ Hf Hnth)
    as (loaded' & Hnth' & from & Hcons & Hused).
  rewrite Hnth'.
  pose proof (block_field_not_local is_local Hlocal (Z.to_nat fld) Value)
    as Hl.
  pose proof (field_used_read is_local G sol Hc a base (Node_variable x) from
                _ Hl Hcons Hacc Ha Hx) as Hu.
  pose proof (rel_read G sol keep (exact = true) is_local Hc _ _ _ _ _ _ _ Hl
                Hcons Hacc Ha (Hused Hu)) as Hrel'.
  exists loaded'.
  assert (Hcond :
    match fk with
    | Any_value => kind_eqb (kind_of_value loaded') Value
    | Immediate => is_tagged_immediate loaded'
    end
    = match fk with
      | Any_value => kind_eqb (kind_of_value loaded) Value
      | Immediate => is_tagged_immediate loaded
      end).
  { destruct fk;
      [ rewrite (rel_kind _ _ _ _ _ _ _ Hrel')
      | rewrite (rel_tagged _ _ _ _ _ _ _ Hrel') ];
      reflexivity. }
  rewrite Hcond.
  destruct (match fk with
            | Any_value => kind_eqb (kind_of_value loaded) Value
            | Immediate => is_tagged_immediate loaded
            end);
    [ | discriminate Hev ].
  injection Hev as <-. split; [ reflexivity | exact Hrel' ].
Qed.

(* [Is_int] of a variant, whose argument has a use. *)
Lemma is_int_rel (statics statics' : static_env) (v1 v1' : value)
    (n : node) :
  statics_rel statics statics' ->
  rel v1 n v1' -> hu n ->
  eval_is_int statics' true v1' = eval_is_int statics true v1.
Proof.
  intros Hst Hrel Hn. unfold eval_is_int.
  rewrite (rel_tagged _ _ _ _ _ _ _ Hrel).
  destruct (is_tagged_immediate v1); [ reflexivity | ].
  destruct (block_of statics v1) as [[tag fields] | ] eqn:Hb.
  - destruct (block_of_rel statics statics' v1 v1' n tag fields Hst Hrel Hn Hb)
      as (a & fields' & Hb' & _).
    rewrite Hb'. reflexivity.
  - rewrite (block_of_none_rel statics statics' v1 v1' n Hst Hrel Hb).
    reflexivity.
Qed.

(* [Get_tag], whose argument has a use. *)
Lemma get_tag_rel (statics statics' : static_env) (v1 v1' v : value)
    (n : node) :
  statics_rel statics statics' ->
  rel v1 n v1' -> hu n ->
  eval_unary_primitive statics Get_tag v1 = Some v ->
  eval_unary_primitive statics' Get_tag v1' = Some v.
Proof.
  intros Hst Hrel Hn Hev. cbn [eval_unary_primitive] in Hev |- *.
  destruct (block_of statics v1) as [[tag fields] | ] eqn:Hb;
    [ | discriminate Hev ].
  destruct (block_of_rel statics statics' v1 v1' n tag fields Hst Hrel Hn Hb)
    as (a & fields' & Hb' & _).
  rewrite Hb'. exact Hev.
Qed.

(* The unboxing of a number, into a variable with a use: the number is
   used, and kept. *)
Lemma unbox_rel (statics statics' : static_env) (x : variable) (base : node)
    (bn : boxable_number) (v1 v1' v : value) :
  statics_rel statics statics' ->
  g_accessor G (Node_variable x) (Field_boxed_number bn) base ->
  rel v1 base v1' -> hu (Node_variable x) ->
  eval_unary_primitive statics (Unbox_number bn) v1 = Some v ->
  eval_unary_primitive statics' (Unbox_number bn) v1' = Some v.
Proof.
  intros Hst Hacc Hrel Hx Hev. cbn [eval_unary_primitive] in Hev |- *.
  destruct (boxed_number_of statics v1) as [c | ] eqn:Hbx;
    [ | discriminate Hev ].
  destruct (boxed_number_of_cases _ _ _ Hbx) as [-> | (s & -> & Hl)].
  - destruct Hrel as (a & bn' & from & Ha & Hk & Hcons & ->).
    destruct (is_number_of_kind bn c) eqn:Hkind; [ | discriminate Hev ].
    pose proof (is_number_of_kind_unique _ _ _ Hk Hkind) as Heq.
    subst bn'.
    rewrite (field_used_read is_local G sol Hc a base (Node_variable x) from
               _ (boxed_field_not_local is_local Hlocal bn) Hcons Hacc Ha Hx).
    change (boxed_number_of statics' (Value_boxed_number c))
      with (boxed_number_of statics (Value_boxed_number c)).
    rewrite Hbx. cbv beta iota. rewrite Hkind. exact Hev.
  - destruct (proj1 (proj2 (proj2 Hst)) s _ Hl)
      as [(tag & fields & Heq) | (slot & syms & decls & vs & Heq & _)];
      discriminate Heq.
Qed.

(* The boxing of a number: kept with its contents if the field of the
   number is used, poison otherwise. *)
Lemma box_number_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (bn : boxable_number)
    (s1 : simple) (v1 v : value) :
  env_ok (env_kinds en) vars vars' -> env_flows_ vars ->
  g_constructor G (Node_variable x) (Field_boxed_number bn)
    (node_of_simple s1) ->
  simple_noted noted s1 ->
  eval_simple vars s1 = Some v1 ->
  eval_unary_primitive statics (Box_number bn) v1 = Some v ->
  ∃ v' : value,
    eval_named statics' vars'
      (rebuild_named sol en x (Prim (Unary (Box_number bn)) [s1]))
    = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hvars Hcons Hnoted Hs1 Hev. cbn [eval_unary_primitive] in Hev.
  destruct v1 as [c | | | | | | | | ]; try discriminate Hev.
  destruct (is_number_of_kind bn c) eqn:Hk; [ | discriminate Hev ].
  injection Hev as <-.
  pose proof (simple_flows is_local G sol noted cds keep Hc Hconsts Hsyms _ _ _
                Hvars Hnoted Hs1) as Hflows1.
  assert (Hself : may_come_from sol (Node_variable x) (Node_variable x)).
  { exact (alloc_self is_local G sol Hc _ _ _ Hcons
             (as_hs _ _ _ Hc _ Hflows1)). }
  cbn [rebuild_named]. unfold field_used.
  destruct (s_field_used sol (Node_variable x) (Field_boxed_number bn))
    eqn:Hused.
  - pose proof (field_used_hu G sol is_local Hc Hexact _ _ _
                  (boxed_field_not_local is_local Hlocal bn) Hused Hcons)
      as Hhu.
    destruct (rewrite_simple_eval en vars vars' s1 _ Henv Hnoted Hs1)
      as (v1' & Hs1' & _ & Hkept).
    destruct (Hkept (kept_of_hu en s1 Hhu)) as [Hr Hrel].
    destruct Hrel as [_ ->]. rewrite Hr in Hs1'.
    exists (Value_boxed_number c). split.
    + rewrite (unary_target en vars' statics' (Box_number bn) s1 _ Hr Hs1').
      cbn [eval_unary_primitive]. rewrite Hk. reflexivity.
    + exists (Node_variable x), bn, (node_of_simple s1). rewrite Hused. auto.
  - exists (Value_const (Poison Value)). split; [ reflexivity | ].
    exists (Node_variable x), bn, (node_of_simple s1). rewrite Hused. auto.
Qed.

(* A block made: its fields used are related. *)
Lemma make_block_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (tag : Z)
    (fields : list simple) (vs : list value) (v : value) :
  env_ok (env_kinds en) vars vars' ->
  statics_rel statics statics' ->
  block_edges G noted (Node_variable x) fields ->
  eval_simples vars fields = Some vs ->
  eval_variadic_primitive statics (Make_block (Block_kind_values tag)) vs
  = Some v ->
  ∃ v' : value,
    eval_named statics' vars'
      (Prim (Variadic (Make_block (Block_kind_values tag)))
         (rebuild_fields sol en (Node_variable x) 0 fields))
    = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hst (Hcons & His & Hgt & Hnoted) Hvs Hev.
  cbn [eval_variadic_primitive] in Hev.
  destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs)
    eqn:Hkinds; [ | discriminate Hev ].
  destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
    [ | discriminate Hev ].
  injection Hev as <-.
  destruct (rebuild_fields_eval en vars vars' (Node_variable x) Henv fields 0
              vs Hcons Hnoted Hvs)
    as (vs' & Hvs' & Hf & Hpoint).
  rewrite forallb_forall in Hkinds, Hbound.
  exists (Value_block tag vs'). split.
  - cbn [eval_named]. rewrite Hvs'. cbv beta iota.
    rewrite eval_prim_variadic. cbn [eval_variadic_primitive].
    assert (Hkinds' : forallb (fun v => kind_eqb (kind_of_value v) Value) vs'
                      = true).
    { apply forallb_forall. intros v' Hin. apply In_nth_error in Hin as [i Hi].
      destruct (fields_related_nth' _ _ _ _ _ _ Hf Hi) as (w & Hw & _).
      destruct (Hpoint i w v' Hw Hi) as [-> | (n & Hrel & _)];
        [ reflexivity | ].
      rewrite (rel_kind _ _ _ _ _ _ _ Hrel).
      exact (Hkinds w (nth_error_In _ _ Hw)). }
    assert (Hbound' : forallb (bound_if_symbol statics') vs' = true).
    { apply forallb_forall. intros v' Hin. apply In_nth_error in Hin as [i Hi].
      destruct (fields_related_nth' _ _ _ _ _ _ Hf Hi) as (w & Hw & _).
      destruct (Hpoint i w v' Hw Hi) as [-> | (n & Hrel & Hn)];
        [ reflexivity | ].
      exact (bound_if_symbol_rel statics statics' w v' n Hst Hrel Hn
               (Hbound w (nth_error_In _ _ Hw))). }
    rewrite Hkinds', Hbound'. reflexivity.
  - exists (Node_variable x), vs'.
    split; [ | split; [ reflexivity | exact Hf ] ].
    exact (alloc_self is_local G sol Hc _ _ _ His
             (as_hs _ _ _ Hc _ (consts_any_source is_local G sol Hc Hconsts))).
Qed.
(* A simple that is a constant or has any usage: in the second run, a
   value [sim] to it, and related to it at a variable that has any source
   and uses the simple. *)
Definition element_rel (w w' : value) : Prop :=
  (w' = w
   ∧ ((∃ c : const, w = Value_const c)
      ∨ ∃ s : symbol, w = Value_symbol s ∧ au (Node_symbol s)))
  ∨ ∃ m : node, au m ∧ rel w m w'.

Lemma element_rel_at (w w' : value) (x : variable) :
  element_rel w w' -> as_ (Node_variable x) -> rel w (Node_variable x) w'.
Proof.
  intros [(-> & [(c & ->) | (s & -> & Hs)]) | (m & Hm & Hrel)] Hx.
  - cbn [rel]. split; [ exact Hx | reflexivity ].
  - cbn [rel]. split; [ reflexivity | split ].
    + exact (may_come_from_any is_local G sol Hc _ _ Hx (fun _ => Hs)).
    + intros _. exact (au_hu _ _ _ Hc _ Hs).
  - exact (rel_escape G sol keep (exact = true) is_local Hc w _ _ w' Hx
             (fun _ => Hm) Hrel).
Qed.

Lemma element_rel_sim (w w' : value) : element_rel w w' -> sim w w'.
Proof.
  intros [(-> & [(c & ->) | (s & -> & Hs)]) | (m & Hm & Hrel)].
  - reflexivity.
  - split; [ reflexivity | exact Hs ].
  - exact (rel_au_sim G sol keep (exact = true) is_local Hc Hlocal _ _ _ Hm
             Hrel).
Qed.

Lemma rewrite_simple_au (en : rebuild_env) (vars vars' : env) (s : simple)
    (w : value) :
  env_ok (env_kinds en) vars vars' ->
  (∃ c : const, s = Const c) ∨ au (node_of_simple s) ->
  eval_simple vars s = Some w ->
  ∃ w' : value,
    eval_simple vars' (rewrite_simple sol en s) = Some w'
    ∧ element_rel w w'.
Proof.
  intros Henv Hs Hw. destruct s as [y | sym | c]; simpl in Hw, Hs.
  - destruct Hs as [(c & Heq) | Hau]; [ discriminate Heq | ].
    destruct (Henv y w Hw) as [_ Hused].
    unfold rewrite_simple. cbv zeta.
    rewrite (env_ok_kind en vars vars' y w Henv Hw).
    rewrite (is_var_used_hu y _ (au_hu _ _ _ Hc _ Hau)).
    destruct (Hused (is_var_used_hu y _ (au_hu _ _ _ Hc _ Hau)))
      as (w' & Hl & Hrel).
    exists w'. split; [ exact Hl | right; exists (Node_variable y); auto ].
  - destruct Hs as [(c & Heq) | Hau]; [ discriminate Heq | ].
    unfold rewrite_simple, is_symbol_used. rewrite (au_hu _ _ _ Hc _ Hau).
    exists w. split; [ exact Hw | ]. injection Hw as <-.
    left. split; [ reflexivity | right; exists sym; auto ].
  - exists w. split; [ exact Hw | ]. injection Hw as <-.
    left. split; [ reflexivity | left; exists c; reflexivity ].
Qed.

Lemma rewrite_simples_au (en : rebuild_env) (vars vars' : env) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (args : list simple) (vs : list value),
    (∀ s : simple, s ∈ args ->
       (∃ c : const, s = Const c) ∨ au (node_of_simple s)) ->
    eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' (map (rewrite_simple sol en) args) = Some vs'
      ∧ Forall2 element_rel vs vs'.
Proof.
  intros Henv. induction args as [ | s args IH]; intros vs Hargs Hvs;
    simpl in Hvs |- *.
  - injection Hvs as <-. exists []. split; [ reflexivity | constructor ].
  - destruct (eval_simple vars s) as [w | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
    injection Hvs as <-.
    destruct (rewrite_simple_au en vars vars' s w Henv
                (Hargs s (or_introl eq_refl)) Hs) as (w' & Hs' & Hw).
    destruct (IH ws (fun s' Hin => Hargs s' (or_intror Hin)) eq_refl)
      as (ws' & Hws' & Hall).
    exists (w' :: ws'). rewrite Hs', Hws'.
    split; [ reflexivity | constructor; assumption ].
Qed.

Lemma Forall2_values_related {R : value -> value -> Prop} (l l' : list value) :
  Forall2 R l l' -> values_related R l l'.
Proof. induction 1; simpl; auto. Qed.

Lemma values_related_Forall2 {R : value -> value -> Prop} (l l' : list value) :
  values_related R l l' -> Forall2 R l l'.
Proof.
  revert l'. induction l as [ | w l IH]; intros [ | w' l'] H; simpl in H;
    try contradiction; constructor;
    [ exact (proj1 H) | exact (IH _ (proj2 H)) ].
Qed.

Ltac obs_close :=
  simpl; intros;
  first [ discriminate | reflexivity | congruence
        | (match goal with
           | H : context [lookup ?a ?b] |- _ => destruct (lookup a b)
           end; congruence) ].

(* [sim] relates what the primitives observe of a value, as
   FlambdaPrimitiveSpec.eval_prim_related asks. *)
Lemma sim_respects (statics statics' : static_env) :
  statics_rel statics statics' -> respects_observations sim statics statics'.
Proof.
  intros Hst. split; [ split | split; [ split; [ | split ] | ] ].
  - intros c v' H. exact H.
  - intros v c H.
    destruct v as [c' | code slots | slot decls vs | s | tag fields | elements
                  | c' | l | callee args arity];
      cbn [sim] in H; try contradiction.
    + exact (eq_sym H).
    + destruct H as (vs' & Heq & _). discriminate Heq.
    + destruct H as [Heq _]. discriminate Heq.
    + destruct H as (fields' & Heq & _). discriminate Heq.
    + destruct H as (elements' & Heq & _). discriminate Heq.
    + discriminate H.
    + discriminate H.
  - intros c. split; reflexivity.
  - intros tag fields fields' H. exists fields'.
    split; [ reflexivity | exact (Forall2_values_related _ _ H) ].
  - intros elements elements' H. exists elements'.
    split; [ reflexivity | exact (Forall2_values_related _ _ H) ].
  - intros v v' Hsim.
    destruct v as [c | code slots | slot decls vs | s | tag fields | elements
                  | c | l | callee args arity];
      cbn [sim] in Hsim; try contradiction.
    + subst v'. repeat split; obs_close.
    + destruct Hsim as (vs' & -> & _). repeat split; obs_close.
    + destruct Hsim as [-> Hau].
      destruct (statics_au statics statics' s Hst Hau)
        as (Hnone & Hblk & Hclo & Honly).
      destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl.
      * destruct (Honly allocated eq_refl)
          as [(t0 & f0 & ->) | (sl & sy & de & vs0 & ->)].
        -- destruct (Hblk t0 f0 eq_refl) as (fields' & Hl' & Hf).
           repeat split; simpl; rewrite ?Hl, ?Hl'; try obs_close.
           intros t f Hb. injection Hb as <- <-. exists fields'.
           split; [ reflexivity | exact (values_related_Forall2 _ _ Hf) ].
        -- destruct (Hclo sl sy de vs0 eq_refl) as (vs' & Hl').
           repeat split; simpl; rewrite ?Hl, ?Hl'; obs_close.
      * pose proof (Hnone eq_refl) as Hl'.
        repeat split; simpl; rewrite ?Hl, ?Hl'; obs_close.
    + destruct Hsim as (fields' & -> & Hf).
      repeat split; try obs_close.
      intros t f Hb. simpl in Hb. injection Hb as <- <-. exists fields'.
      split; [ reflexivity | exact (values_related_Forall2 _ _ Hf) ].
    + destruct Hsim as (elements' & -> & Hf).
      repeat split; try obs_close.
      intros e Ha. simpl in Ha. injection Ha as <-. exists elements'.
      split; [ reflexivity | exact (values_related_Forall2 _ _ Hf) ].
    + subst v'. repeat split; obs_close.
    + subst v'. repeat split; obs_close.
Qed.

(* The primitives of the default case: a constant, the identity, a load
   from an immutable array and the making of one. *)
Lemma default_prim_cases (statics : static_env) (p : prim)
    (args : list simple) (vs : list value) (v : value) :
  prim_classified p args = false ->
  length vs = length args ->
  eval_prim statics p vs = Some v ->
  (∃ c : const, v = Value_const c)
  ∨ (∃ k : flambda_kind, p = Unary (Opaque_identity k) ∧ vs = [v])
  ∨ (∃ (ak : array_kind) (lk : array_load_kind) (mut : mutability)
       (arr idx : value) (elements : list value),
       p = Binary (Array_load ak lk mut) ∧ vs = [arr; idx]
       ∧ array_of statics arr = Some elements
       ∧ array_loaded lk elements idx = Some v)
  ∨ (∃ ak : array_kind,
       p = Variadic (Make_immutable_array ak)
       ∧ v = Value_immutable_array vs).
Proof.
  intros Hcl Hlen Heval.
  destruct (eval_prim_arity _ _ _ _ Heval)
    as [(p' & v1 & -> & -> & Hu) | [(p' & v1 & v2 & -> & -> & Hb)
                                  | (p' & -> & Hv)]].
  - destruct args as [ | s1 [ | s2 args]]; simpl in Hlen; try discriminate Hlen.
    destruct p' as [kind mut fld | [ | ] | | | ak | | ok | ik op | src dst
                   | | bn | bn | | | r];
      simpl in Hcl; try discriminate Hcl; cbn [eval_unary_primitive] in Hu;
      try (const_value Hu).
    destruct (kind_eqb (kind_of_value v1) ok && bound_if_symbol statics v1);
      [ | discriminate Hu ].
    injection Hu as ->. right. left. exists ok. auto.
  - destruct p' as [ak lk mut' | width | op | ik op | ik op | ik beh];
      cbn [eval_binary_primitive] in Hb; try (const_value Hb).
    destruct (array_of statics v1) as [elements | ] eqn:Harr;
      [ | discriminate Hb ].
    right. right. left. exists ak, lk, mut', v1, v2, elements. auto.
  - destruct p' as [bk | ak]; [ discriminate Hcl | ].
    cbn [eval_variadic_primitive] in Hv.
    destruct (array_made ak vs) as [elements | ] eqn:Hmade;
      [ | discriminate Hv ].
    destruct (array_made_of_arguments _ _ _ Hmade) as [-> _].
    destruct (forallb (bound_if_symbol statics) vs); [ | discriminate Hv ].
    injection Hv as <-. right. right. right. exists ak. auto.
Qed.

Lemma sim_const_r (v : value) (c : const) :
  sim v (Value_const c) -> v = Value_const c.
Proof.
  destruct v; cbn [sim]; try contradiction; try (intros H; discriminate H);
    try (intros [H _]; discriminate H);
    try (intros (? & H & _); discriminate H).
  intros H. exact (eq_sym H).
Qed.

Lemma values_related_length {R : value -> value -> Prop}
    (l l' : list value) :
  values_related R l l' -> length l' = length l.
Proof.
  revert l'. induction l as [ | w l IH]; intros [ | w' l'] H; simpl in H;
    try contradiction; [ reflexivity | ].
  simpl. f_equal. exact (IH _ (proj2 H)).
Qed.

Lemma values_related_nth {R : value -> value -> Prop} (l l' : list value)
    (i : nat) (w w' : value) :
  values_related R l l' -> nth_error l i = Some w ->
  nth_error l' i = Some w' -> R w w'.
Proof.
  revert l' i. induction l as [ | a0 l IH]; intros [ | a0' l'] i H Hn Hn';
    simpl in H; try contradiction; [ destruct i; discriminate | ].
  destruct i as [ | i]; simpl in Hn, Hn'.
  - injection Hn as <-. injection Hn' as <-. exact (proj1 H).
  - exact (IH l' i (proj2 H) Hn Hn').
Qed.

(* The loads at one index from two arrays of related elements. *)
Lemma array_loaded_related (R : value -> value -> Prop)
    (lk : array_load_kind) (els els' : list value) (idx v v' : value) :
  values_related R els els' ->
  array_loaded lk els idx = Some v -> array_loaded lk els' idx = Some v' ->
  R v v'.
Proof.
  intros Hr Hl Hl'. unfold array_loaded, array_index in Hl, Hl'.
  rewrite (values_related_length _ _ Hr) in Hl'.
  destruct (int_of_value Standard_tagged_immediate idx) as [i | ];
    [ | discriminate Hl ].
  cbv beta iota in Hl, Hl'.
  destruct ((0 <=? i)%Z && (i <? Z.of_nat (length els))%Z);
    [ | discriminate Hl ].
  cbv beta iota in Hl, Hl'.
  destruct (nth_error els (Z.to_nat i)) as [w | ] eqn:Hn;
    [ | discriminate Hl ].
  destruct (nth_error els' (Z.to_nat i)) as [w' | ] eqn:Hn';
    [ | discriminate Hl' ].
  cbv beta iota in Hl, Hl'.
  destruct (array_element_ok _ w); [ injection Hl as <- | discriminate Hl ].
  destruct (array_element_ok _ w'); [ injection Hl' as <- | discriminate Hl' ].
  exact (values_related_nth _ _ _ _ _ Hr Hn Hn').
Qed.

(* A primitive of the default case, whose variable has a use: its
   arguments have any usage, so the second run gives it [sim] values, and
   it has a related value. *)
Lemma default_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (p : prim)
    (args : list simple) (vs : list value) (v : value) :
  env_ok (env_kinds en) vars vars' ->
  statics_rel statics statics' ->
  default_prim_edges G x args -> prim_classified p args = false ->
  hu (Node_variable x) ->
  eval_simples vars args = Some vs -> eval_prim statics p vs = Some v ->
  ∃ v' : value,
    eval_named statics' vars' (Prim p (map (rewrite_simple sol en) args))
    = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hst [Has Huse] Hcl Hx Hvs Hev.
  pose proof (any_source_input _ _ _ Hc _ Has) as Hasx.
  assert (Hargs : ∀ s : simple, s ∈ args ->
            (∃ c : const, s = Const c) ∨ au (node_of_simple s)).
  { intros s Hin. destruct s as [y | sym | c]; [ right | right | left ].
    - exact (any_usage_use _ _ _ Hc _ _ Hx (Huse _ (free_names_var _ _ Hin))).
    - exact (any_usage_use _ _ _ Hc _ _ Hx
               (Huse _ (free_names_symbol _ _ Hin))).
    - exists c. reflexivity. }
  destruct (rewrite_simples_au en vars vars' Henv args vs Hargs Hvs)
    as (vs' & Hvs' & Hall).
  assert (Hsims : Forall2 sim vs vs')
    by exact (Forall2_impl _ element_rel_sim Hall).
  destruct (eval_prim_related sim statics statics' p vs vs' v
              (sim_respects statics statics' Hst) Hsims Hev)
    as (v' & Hev' & Hsim).
  exists v'. cbn [eval_named]. rewrite Hvs'. split; [ exact Hev' | ].
  pose proof (eval_simples_length _ _ _ Hvs) as Hlen.
  assert (Hlen' : length vs' = length args)
    by (rewrite <- (Forall2_length Hall); exact Hlen).
  destruct (default_prim_cases statics' p args vs' v' Hcl Hlen' Hev')
    as [(c' & ->) | Ht].
  { rewrite (sim_const_r _ _ Hsim). cbn [rel].
    split; [ exact Hasx | reflexivity ]. }
  destruct (default_prim_cases statics p args vs v Hcl Hlen Hev)
    as [(c & ->) | [(k & -> & ->) | [(ak & lk & mut & arr & idx & elements
                                       & -> & -> & Harr & Hload)
                                     | (ak & -> & ->)]]].
  - cbn [sim] in Hsim. subst v'. cbn [rel].
    split; [ exact Hasx | reflexivity ].
  - destruct Ht as [(k' & _ & ->) | [(ak' & lk' & mut' & arr' & idx'
                                     & elements' & Hp & _) | (ak' & Hp & _)]];
      [ | discriminate Hp | discriminate Hp ].
    inversion Hall as [ | w w' ws ws' Hw _ ]; subst.
    exact (element_rel_at _ _ x Hw Hasx).
  - destruct Ht as [(k' & Hp & _) | [(ak' & lk' & mut' & arr' & idx'
                                     & elements' & Hp & -> & Harr' & Hload')
                                   | (ak' & Hp & _)]];
      [ discriminate Hp | | discriminate Hp ].
    injection Hp as <- <- <-.
    inversion Hall as [ | ? ? ? ? Harr_rel Hrest ]; subst.
    inversion Hrest as [ | ? ? ? ? Hidx_rel _ ]; subst.
    assert (Hidx : idx' = idx).
    { pose proof (element_rel_sim _ _ Hidx_rel) as Hs.
      destruct idx as [ci | | | | | | | | ];
        try (unfold array_loaded, array_index in Hload; simpl in Hload;
             discriminate Hload).
      exact Hs. }
    subst idx'.
    destruct arr as [ | | | s | | els | | | ]; simpl in Harr;
      try discriminate Harr.
    + destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl;
        [ | discriminate Harr ].
      destruct (proj1 (proj2 (proj2 Hst)) s _ Hl)
        as [(t & f & ->) | (sl & sy & de & vs0 & -> & _)]; discriminate Harr.
    + injection Harr as <-.
      pose proof (element_rel_at _ _ x Harr_rel Hasx) as Hr.
      cbn [rel] in Hr. destruct Hr as (_ & elements'' & -> & Hels).
      simpl in Harr'. injection Harr' as <-.
      exact (element_rel_at _ _ x
               (array_loaded_related _ lk _ _ _ _ _ Hels Hload Hload') Hasx).
  - destruct Ht as [(k' & Hp & _) | [(ak' & lk' & mut' & arr' & idx'
                                     & elements' & Hp & _) | (ak' & _ & ->)]];
      [ discriminate Hp | discriminate Hp | ].
    cbn [rel]. split; [ exact Hasx | ]. exists vs'.
    split; [ reflexivity | exact (Forall2_values_related _ _ Hall) ].
Qed.

(* A [Let] of kind [Region] or [Rec_info], which the rebuild keeps: its
   value is a constant, of a simple or of the identity, the same in the
   two runs. *)
Lemma region_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (d : named) (v : value) :
  env_ok (env_kinds en) vars vars' ->
  named_closure_free d = true -> named_edges_ x d ->
  eval_named statics vars d = Some v ->
  kind_of_value v = Region ∨ kind_of_value v = Rec_info ->
  flows_ v (Node_variable x) ->
  ∃ v' : value,
    eval_named statics' vars' (rebuild_named sol en x d) = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hfrag Hedges Hev Hk Hflows.
  destruct (not_value_const v (region_not_value _ Hk)) as (c & ->).
  exists (Value_const c). split; [ | split; [ exact Hflows | reflexivity ] ].
  destruct d as [s | p args | | | | | ]; simpl in Hfrag; try discriminate Hfrag;
    simpl in Hedges.
  - destruct Hedges as [_ Hnoted]. cbn [eval_named rebuild_named] in Hev |- *.
    destruct (rewrite_simple_eval en vars vars' s _ Henv Hnoted Hev)
      as (v' & Hs' & _ & Hkept).
    destruct (Hkept (kept_of_region en vars vars' s _ Henv Hev Hk))
      as [_ [_ ->]].
    exact Hs'.
  - cbn [eval_named] in Hev.
    destruct (eval_simples vars args) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    destruct (eval_prim_kinds _ _ _ _ Hev) as [_ Hres].
    pose proof Hk as Hkp. rewrite Hres in Hkp.
    destruct (region_prim p Hkp) as (k & ->).
    destruct (eval_prim_arity _ _ _ _ Hev)
      as [(up & v1 & Hp & -> & Hu) | [(bp & v1 & v2 & Hp & _) | (vp & Hp & _)]];
      [ | discriminate Hp | discriminate Hp ].
    injection Hp as <-.
    destruct args as [ | s1 [ | s2 args]]; simpl in Hvs;
      try (destruct (eval_simple vars s1); [ | discriminate Hvs ];
           destruct (eval_simple vars s2); [ | discriminate Hvs ];
           destruct (eval_simples vars args); discriminate Hvs);
      [ discriminate Hvs | ].
    destruct (eval_simple vars s1) as [w | ] eqn:Hs1; [ | discriminate Hvs ].
    injection Hvs as <-.
    cbn [eval_unary_primitive] in Hu.
    destruct (kind_eqb (kind_of_value w) k && bound_if_symbol statics w)
      eqn:Hcond; [ | discriminate Hu ].
    injection Hu as ->.
    assert (Hnoted : simple_noted noted s1).
    { destruct s1; simpl; try exact I.
      simpl in Hs1. injection Hs1 as Heq. discriminate Heq. }
    destruct (rewrite_simple_eval en vars vars' s1 _ Henv Hnoted Hs1)
      as (w' & Hs1' & _ & Hkept).
    destruct (Hkept (kept_of_region en vars vars' s1 _ Henv Hs1 Hk))
      as [Hr [_ ->]].
    rewrite Hr in Hs1'. cbn [rebuild_named].
    rewrite (unary_target en vars' statics' (Opaque_identity k) s1 _ Hr Hs1').
    cbn [eval_unary_primitive].
    apply andb_true_iff in Hcond as [Hkind _]. rewrite Hkind. reflexivity.
Qed.

(* The projection of a function slot from a closure of a set, related at
   the variable that the projection binds. *)
Lemma rel_project_function_slot (slot o : function_slot)
    (decls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)) (base y : node) (id0 id1 : code_id) :
  is_local (Field_function_slot o) = false ->
  lookup slot decls = Some id0 -> lookup o decls = Some id1 ->
  g_accessor G y (Field_function_slot o) base ->
  rel (Value_closure_of_set slot decls vs) base
    (Value_closure_of_set slot decls vs') ->
  rel (Value_closure_of_set o decls vs) y (Value_closure_of_set o decls vs').
Proof.
  intros Hl Hl0 Hl1 Hacc Hrel. cbn [rel] in Hrel.
  destruct Hrel as (sites & vs'' & Hmay & Heq & Hex & Hall).
  injection Heq as <-.
  destruct (Hall slot id0 Hl0) as (_ & Hfs & _).
  apply (rel_read G sol keep (exact = true) is_local Hc _ _ _ _ _ _ _ Hl
           (Hfs o id1 Hl1) Hacc Hmay).
  cbn [rel]. exists sites, vs'.
  split; [ exact (proj1 (Hall o id1 Hl1)) | ].
  split; [ reflexivity | split; [ exact Hex | exact Hall ] ].
Qed.

(* The value of a [Let] of a projection, that the rebuild keeps: a
   function slot of a closure, of the same set in the two runs; a value
   slot used, kept by the rebuild. *)
Lemma closure_named_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (d : named) (v : value) :
  env_ok (env_kinds en) vars vars' ->
  statics_rel statics statics' -> env_flows_ vars -> statics_flow_ statics ->
  named_no_code (kind_of_value v) d = true -> named_plain d = true ->
  named_closure_free d = false ->
  named_edges_ x d ->
  eval_named statics vars d = Some v ->
  is_var_used sol x (kind_of_value v) = true ->
  ∃ v' : value,
    eval_named statics' vars' (rebuild_named sol en x d) = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hst Hvars Hsflow Hnc Hpl Hcf Hedges Hev Hused.
  destruct d as [s | p args | code vs | slot decls vs | w c | pf w c
                | mf mt c];
    simpl in Hpl, Hcf; try discriminate Hpl; try discriminate Hcf;
    [ simpl in Hnc; discriminate Hnc | | ].
  - (* [Project_value_slot_from] *)
    assert (Hx : hu (Node_variable x)).
    { simpl in Hnc. destruct (kind_of_value v); try discriminate Hnc;
        exact Hused. }
    simpl in Hedges. destruct Hedges as ((Hacc & Hnoted) & Hk & Hslots).
    pose proof (Hslots (Field_value_slot w)) as Hl.
    cbn [eval_named rebuild_named] in Hev |- *.
    destruct (eval_simple vars c) as [vc | ] eqn:Hvc; [ | discriminate Hev ].
    pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc) as Hc_hu.
    destruct (rewrite_simple_eval en vars vars' c vc Henv Hnoted Hvc)
      as (vc' & Hs' & _ & Hkept).
    destruct (Hkept (kept_of_hu en c Hc_hu)) as [Hr Hrel].
    rewrite Hr in Hs' |- *. rewrite Hs'. cbv beta iota.
    apply project_value_slot_defined in Hev as (decls & slots & Hcis & Hw).
    destruct vc as [ | code slots0 | slot0 decls0 slots0 | s | | | | | ];
      simpl in Hcis; try discriminate Hcis.
    + cbn [rel] in Hrel. destruct Hrel.
    + destruct (lookup slot0 decls0) as [id0 | ] eqn:Hl0;
        [ | discriminate Hcis ].
      injection Hcis as -> -> ->.
      cbn [rel] in Hrel. destruct Hrel as (sites & vs' & Hmay & -> & _ & Hall).
      destruct (Hall pf id0 Hl0) as (_ & _ & Hsr).
      destruct (proj1 (slots_related_iff _ _) Hsr w v Hw Hk)
        as (from & Hcons & Hfu).
      pose proof (field_used_read is_local G sol Hc _ _ _ _ _ Hl Hcons Hacc
                    Hmay Hx) as Hused_w.
      destruct (Hfu Hused_w) as (x' & Hx' & Hrelx).
      pose proof (field_used_hu G sol is_local Hc Hexact _ _ _ Hl Hused_w Hcons)
        as Hhu_from.
      exists x'. split.
      * unfold project_value_slot. simpl. rewrite Hl0. simpl.
        rewrite Pos.eqb_refl. exact Hx'.
      * exact (rel_read G sol keep (exact = true) is_local Hc _ _ _ _ _ _ _ Hl
                 Hcons Hacc Hmay (Hrelx Hhu_from)).
    + destruct (lookup s (static_values statics))
        as [[ | slot0 syms0 decls0 slots0 | | | | | ] | ] eqn:Hls;
        try discriminate Hcis.
      * destruct (Hsflow s _ Hls) as [_ []].
      * destruct (lookup slot0 decls0) as [id0 | ] eqn:Hl0;
          [ | discriminate Hcis ].
        injection Hcis as -> -> ->.
        cbn [rel] in Hrel. destruct Hrel as (-> & Hmay & Hhu_s).
        destruct (proj1 (proj2 (proj2 Hst)) s _ Hls)
          as [(t & f & Heq) | (sl & sy & de & vs0 & Heq & Htarget & Hbound)];
          [ discriminate Heq | ].
        injection Heq as <- <- <- <-.
        destruct Htarget as [Hnone | (vs' & Hls' & _ & Hsr)];
          [ destruct (Hbound (Hhu_s Hc_hu) Hnone) | ].
        destruct (proj1 (slots_related_iff _ _) Hsr w v Hw Hk)
          as (from & Hcons & Hfu).
        pose proof (field_used_read is_local G sol Hc _ _ _ _ _ Hl Hcons Hacc
                      Hmay Hx) as Hused_w.
        destruct (Hfu Hused_w) as (x' & Hx' & Hrelx).
        pose proof (field_used_hu G sol is_local Hc Hexact _ _ _ Hl Hused_w
                      Hcons) as Hhu_from.
        exists x'. split.
        -- unfold project_value_slot. simpl. rewrite Hls'. simpl.
           rewrite Hl0. simpl. rewrite Pos.eqb_refl. exact Hx'.
        -- exact (rel_read G sol keep (exact = true) is_local Hc _ _ _ _ _ _ _
                   Hl Hcons Hacc Hmay (Hrelx Hhu_from)).
  - (* [Project_function_slot] *)
    destruct Hedges as ((Hacc & Hnoted) & Hslots).
    cbn [eval_named rebuild_named] in Hev |- *.
    destruct (eval_simple vars c) as [vc | ] eqn:Hvc; [ | discriminate Hev ].
    pose proof (simple_flows is_local G sol noted cds keep Hc Hconsts Hsyms _ _
                  _ Hvars Hnoted Hvc) as Hfc.
    destruct (rewrite_simple_eval en vars vars' c vc Henv Hnoted Hvc)
      as (vc' & Hs' & _ & Hkept).
    unfold project_function_slot in Hev.
    destruct vc as [ | code slots0 | slot0 decls0 slots0 | s | | | | | ];
      simpl in Hev; try discriminate Hev.
    + destruct Hfc.
    + destruct (Pos.eqb_spec slot0 mf) as [-> | _]; [ | discriminate Hev ].
      destruct (lookup mf decls0) as [id0 | ] eqn:Hl0; [ | discriminate Hev ].
      destruct (lookup mt decls0) as [id1 | ] eqn:Hl1; [ | discriminate Hev ].
      injection Hev as <-.
      assert (Hx : hu (Node_variable x)) by exact Hused.
      destruct (Hkept (kept_of_hu en c
                         (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc)))
        as [Hr Hrel].
      rewrite Hr in Hs' |- *.
      pose proof Hrel as Hrel0. cbn [rel] in Hrel0.
      destruct Hrel0 as (sites & vs' & _ & -> & _).
      rewrite Hs'. eexists. split.
      * unfold project_function_slot. simpl.
        rewrite Pos.eqb_refl, Hl0, Hl1. reflexivity.
      * exact (rel_project_function_slot mf mt decls0 slots0 vs' _ _ id0 id1
                 (Hslots _) Hl0 Hl1 Hacc Hrel).
    + destruct (lookup s (static_values statics))
        as [[ | slot0 syms0 decls0 slots0 | | | | | ] | ] eqn:Hls;
        try discriminate Hev.
      * destruct (Hsflow s _ Hls) as [_ []].
      * destruct (Pos.eqb_spec slot0 mf) as [-> | _]; [ | discriminate Hev ].
        destruct (lookup mf decls0) as [id0 | ] eqn:Hl0;
          [ | discriminate Hev ].
        destruct (lookup mt decls0) as [id1 | ] eqn:Hl1;
          [ | discriminate Hev ].
        destruct (lookup mt syms0) as [s' | ] eqn:Hs'0;
          [ | discriminate Hev ].
        injection Hev as <-.
        assert (Hx : hu (Node_variable x)) by exact Hused.
        pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hx Hacc) as Hc_hu.
        destruct (Hkept (kept_of_hu en c Hc_hu)) as [Hr Hrel].
        rewrite Hr in Hs' |- *. cbn [rel] in Hrel.
        destruct Hrel as (-> & Hmay & Hhu).
        destruct (proj1 (proj2 (proj2 Hst)) s _ Hls)
          as [(tag & fields & Heq) | (sl & sy & de & vs0 & Heq & Htarget
                                     & Hbound)];
          [ discriminate Heq | ].
        injection Heq as <- <- <- <-.
        destruct Htarget as [Hnone | (vs' & Hls' & _)];
          [ destruct (Hbound (Hhu Hc_hu) Hnone) | ].
        destruct (Hsflow s _ Hls) as (_ & _ & Hfs & _).
        destruct (Hfs mt s' Hs'0) as [Hcons Hself].
        rewrite Hs'. eexists. split.
        -- unfold project_function_slot. simpl. rewrite Hls'. simpl.
           rewrite Pos.eqb_refl, Hl0, Hl1, Hs'0. reflexivity.
        -- apply (rel_read G sol keep (exact = true) is_local Hc _ _ _ _ _ _ _
                    (Hslots _) Hcons Hacc Hmay).
           cbn [rel]. split; [ reflexivity | split; [ exact Hself | ] ].
           exact (fun h => h).
Qed.

(* The value of a [Let] that the rebuild keeps, in the second run. *)
Lemma named_rel (en : rebuild_env) (vars vars' : env)
    (statics statics' : static_env) (x : variable) (d : named) (v : value) :
  env_ok (env_kinds en) vars vars' ->
  statics_rel statics statics' -> env_flows_ vars -> statics_flow_ statics ->
  named_checked exact (kind_of_value v) d = true -> named_plain d = true ->
  named_edges_ x d ->
  eval_named statics vars d = Some v ->
  flows_ v (Node_variable x) ->
  is_var_used sol x (kind_of_value v) = true ->
  ∃ v' : value,
    eval_named statics' vars' (rebuild_named sol en x d) = Some v'
    ∧ rel v (Node_variable x) v'.
Proof.
  intros Henv Hst Hvars Hsflow Hnc Hpl Hedges Hev Hflows Hused.
  destruct (named_closure_free d) eqn:Hfrag;
    [ | exact (closure_named_rel en vars vars' statics statics' x d v Henv
                 Hst Hvars Hsflow (named_checked_no_code _ _ _ Hnc) Hpl Hfrag
                 Hedges Hev Hused) ].
  destruct (is_var_used_cases x _ Hused) as [Hx | Hk];
    [ | exact (region_rel en vars vars' statics statics' x d v Henv Hfrag
                 Hedges Hev Hk Hflows) ].
  destruct d as [s | p args | | | | | ]; simpl in Hfrag; try discriminate Hfrag;
    simpl in Hedges.
  - (* [Simple] *)
    destruct Hedges as [Hal Hnoted]. cbn [eval_named rebuild_named] in Hev |- *.
    destruct (rewrite_simple_eval en vars vars' s v Henv Hnoted Hev)
      as (v' & Hs' & _ & Hkept).
    pose proof (alias_input _ _ _ Hc _ _ Hal) as Hal'.
    pose proof (has_usage_alias _ _ _ Hc _ _ Hx Hal') as Hs_hu.
    destruct (Hkept (kept_of_hu en s Hs_hu)) as [_ Hrel].
    exists v'. split; [ exact Hs' | ].
    exact (rel_alias G sol keep (exact = true) is_local Hc _ _ _ _ Hal' Hrel).
  - (* [Prim] *)
    cbn [eval_named] in Hev.
    destruct (eval_simples vars args) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    unfold prim_edges in Hedges.
    destruct (prim_classified p args) eqn:Hcl.
    + destruct p as [up | bp | vp | hp].
      * destruct up as [kind mut fld | [ | ] | | | ak | | ok | ik op | src dst
                       | | bn | bn | | | r];
          destruct args as [ | s1 [ | s2 args]]; simpl in Hcl;
          try discriminate Hcl; simpl in Hedges;
          destruct Hedges as [Hedge Hnoted]; simpl in Hvs;
          destruct (eval_simple vars s1) as [v1 | ] eqn:Hs1;
          try discriminate Hvs; injection Hvs as <-;
          rewrite eval_prim_unary in Hev;
          [ | | | | exact (box_number_rel en vars vars' statics statics' x bn
                             s1 v1 v Henv Hvars Hedge Hnoted Hs1 Hev) ];
          pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hx Hedge) as Hs1_hu;
          destruct (rewrite_simple_eval en vars vars' s1 v1 Henv Hnoted Hs1)
            as (v1' & Hs1' & _ & Hkept);
          destruct (Hkept (kept_of_hu en s1 Hs1_hu)) as [Hr Hrel];
          rewrite Hr in Hs1'; cbn [rebuild_named];
          rewrite (unary_target en vars' statics' _ s1 v1' Hr Hs1').
        -- (* [Block_load] *)
           exact (block_load_rel statics statics' x _ kind fld v1 v1' v Hst
                    Hedge Hrel Hx Hev).
        -- (* [Is_int] *)
           cbn [eval_unary_primitive] in Hev |- *.
           rewrite (is_int_rel statics statics' v1 v1' _ Hst Hrel Hs1_hu).
           exists v. split; [ exact Hev | ].
           destruct (eval_is_int statics true v1); [ | discriminate Hev ].
           injection Hev as <-. split; [ exact Hflows | reflexivity ].
        -- (* [Get_tag] *)
           exists v.
           split; [ exact (get_tag_rel statics statics' v1 v1' v _ Hst Hrel
                             Hs1_hu Hev) | ].
           cbn [eval_unary_primitive] in Hev.
           destruct (block_of statics v1) as [[tag fields] | ];
             [ | discriminate Hev ].
           injection Hev as <-. split; [ exact Hflows | reflexivity ].
        -- (* [Unbox_number] *)
           exists v.
           split; [ exact (unbox_rel statics statics' x _ bn v1 v1' v Hst
                             Hedge Hrel Hx Hev) | ].
           cbn [eval_unary_primitive] in Hev.
           destruct (boxed_number_of statics v1) as [c | ];
             [ | discriminate Hev ].
           destruct (is_number_of_kind bn c); [ | discriminate Hev ].
           injection Hev as <-. split; [ exact Hflows | reflexivity ].
      * discriminate Hcl.
      * destruct vp as [[tag] | ak]; [ | discriminate Hcl ].
        rewrite eval_prim_variadic in Hev. cbn [rebuild_named].
        exact (make_block_rel en vars vars' statics statics' x tag args vs v
                 Henv Hst Hedges Hvs Hev).
      * destruct (eval_prim_arity _ _ _ _ Hev)
          as [(up & v1 & Hp & _) | [(bp & v1 & v2 & Hp & _) | (vp & Hp & _)]];
          discriminate Hp.
    + assert (Hplain : rebuild_named sol en x (Prim p args)
                       = Prim p (map (rewrite_simple sol en) args)).
      { destruct (eval_prim_arity _ _ _ _ Hev)
          as [(up & v1 & -> & -> & _) | [(bp & v1 & v2 & -> & -> & _)
                                       | (vp & -> & _)]].
        - destruct up; try reflexivity.
          pose proof (eval_simples_length _ _ _ Hvs) as Hlen.
          destruct args as [ | s1 [ | s2 args]]; simpl in Hlen;
            try discriminate Hlen.
          discriminate Hcl.
        - reflexivity.
        - destruct vp as [[tag] | ak]; [ discriminate Hcl | reflexivity ]. }
      rewrite Hplain.
      exact (default_rel en vars vars' statics statics' x p args vs v Henv
               Hst Hedges Hcl Hx Hvs Hev).
Qed.

(* ------------------------------------------------------------------------ *)
(* Static bindings                                                          *)
(* ------------------------------------------------------------------------ *)

Lemma rebuild_binding_block (en : rebuild_env) (s : symbol) (tag : Z)
    (fields : list simple) :
  rebuild_b en (Bind_symbol s (Static_const_block tag fields))
  = if is_symbol_used sol s
    then [Bind_symbol s
            (Static_const_block tag
               (rebuild_fields sol en (Node_symbol s) 0 fields))]
    else [].
Proof. reflexivity. Qed.

(* A symbol of a static set has a use only if the set is kept. *)
Lemma lookup_existsb (f : symbol -> bool) (slot : function_slot)
    (closure_symbols : list (function_slot * symbol)) (s : symbol) :
  lookup slot closure_symbols = Some s -> f s = true ->
  existsb (fun '(_, sym) => f sym) closure_symbols = true.
Proof.
  induction closure_symbols as [ | [g sym] closure_symbols IH]; simpl;
    [ discriminate | ].
  destruct (Pos.eqb slot g).
  - intros Heq Hf. injection Heq as ->. rewrite Hf. reflexivity.
  - intros Hl Hf. rewrite (IH Hl Hf), orb_true_r. reflexivity.
Qed.

(* A simple of the first run has a value in the second, rewritten. *)
Lemma rewrite_simple_some (en : rebuild_env) (vars vars' : env) (s : simple)
    (v : value) :
  env_ok (env_kinds en) vars vars' ->
  eval_simple vars s = Some v ->
  ∃ v' : value, eval_simple vars' (rewrite_simple sol en s) = Some v'.
Proof.
  intros Henv Hs. destruct s as [x | sym | c]; simpl in Hs.
  - destruct (Henv x v Hs) as [_ Hused].
    unfold rewrite_simple. cbv zeta.
    rewrite (env_ok_kind en vars vars' x v Henv Hs).
    destruct (is_var_used sol x (kind_of_value v)) eqn:Hu.
    + destruct (Hused eq_refl) as (v' & Hl & _). exists v'. exact Hl.
    + eexists. reflexivity.
  - unfold rewrite_simple. destruct (is_symbol_used sol sym); eexists;
      reflexivity.
  - eexists. reflexivity.
Qed.

Lemma lookup_rebuild_value_slots (en : rebuild_env) (bound : list node)
    (value_slots : list (value_slot * simple)) (w : value_slot) (s : simple) :
  lookup w value_slots = Some s ->
  existsb (fun n => s_field_used sol n (Field_value_slot w)) bound = true ->
  lookup w (rebuild_value_slots sol en bound value_slots)
  = Some (rewrite_simple sol en s).
Proof.
  intros Hl Hb. induction value_slots as [ | [w' s'] value_slots IH];
    simpl in Hl |- *; [ discriminate Hl | ].
  destruct (Pos.eqb_spec w w') as [-> | Hne].
  - injection Hl as ->. unfold field_used. rewrite Hb. simpl.
    rewrite Pos.eqb_refl. reflexivity.
  - unfold field_used.
    destruct (existsb (fun n => s_field_used sol n (Field_value_slot w'))
                bound); [ | exact (IH Hl) ].
    simpl. rewrite (proj2 (Pos.eqb_neq w w') Hne). exact (IH Hl).
Qed.

Lemma eval_value_slots_lookup' (vars : env)
    (value_slots : list (value_slot * simple))
    (vs : list (value_slot * value)) (w : value_slot) (s : simple) :
  eval_value_slots vars value_slots = Some vs ->
  lookup w value_slots = Some s ->
  ∃ v : value, lookup w vs = Some v ∧ eval_simple vars s = Some v.
Proof.
  revert vs. induction value_slots as [ | [w' s'] value_slots IH];
    intros vs Hev Hl; simpl in Hev, Hl; [ discriminate Hl | ].
  destruct (eval_simple vars s') as [v' | ] eqn:Hs'; [ | discriminate Hev ].
  destruct (eval_value_slots vars value_slots) as [vs0 | ] eqn:Hvs0;
    [ | discriminate Hev ].
  injection Hev as <-. simpl.
  destruct (Pos.eqb w w').
  - injection Hl as ->. exists v'. auto.
  - exact (IH vs0 eq_refl Hl).
Qed.

(* The value slots of a closure in the second run: those that the rebuild
   keeps ([rebuild_value_slots]), related at a node of [bound] that has
   them used. *)
Lemma value_slots_rel (en : rebuild_env) (vars vars' : env)
    (bound : list node) (value_slots : list (value_slot * simple))
    (vs : list (value_slot * value)) :
  env_ok (env_kinds en) vars vars' ->
  eval_value_slots vars value_slots = Some vs ->
  ∃ vs' : list (value_slot * value),
    eval_value_slots vars' (rebuild_value_slots sol en bound value_slots)
    = Some vs'
    ∧ (value_slots = [] -> vs' = vs)
    ∧ ∀ a : node,
        a ∈ bound ->
        (∀ (w : value_slot) (s : simple),
           lookup w value_slots = Some s -> keep w = true ->
           g_constructor G a (Field_value_slot w) (node_of_simple s)
           ∧ simple_noted noted s) ->
        slots_related (slot_rel a vs') vs [].
Proof.
  intros Henv Hev.
  assert (Hex : ∃ vs' : list (value_slot * value),
            eval_value_slots vars' (rebuild_value_slots sol en bound
                                      value_slots)
            = Some vs').
  { clear - Henv Hev. revert vs Hev.
    induction value_slots as [ | [w s] value_slots IH]; intros vs Hev;
      [ exists []; reflexivity | ].
    simpl in Hev.
    destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate Hev ].
    destruct (eval_value_slots vars value_slots) as [vs0 | ] eqn:Hvs0;
      [ | discriminate Hev ].
    destruct (IH vs0 eq_refl) as (vs0' & Hvs0').
    unfold rebuild_value_slots in Hvs0' |- *. cbn [flat_map].
    destruct (existsb _ bound); [ | exact (ex_intro _ vs0' Hvs0') ].
    destruct (rewrite_simple_some en vars vars' s v Henv Hs) as (v' & Hs').
    cbn [app eval_value_slots]. rewrite Hs', Hvs0'. eexists. reflexivity. }
  destruct Hex as (vs' & Hvs'). exists vs'.
  split; [ exact Hvs' | split ].
  - intros ->. simpl in Hev, Hvs'. injection Hev as <-.
    injection Hvs' as <-. reflexivity.
  - intros a Ha Hcons. apply slots_related_iff. intros w x Hx Hk.
    destruct (eval_value_slots_lookup _ _ _ _ _ Hev Hx) as (s & Hs & Hsx).
    destruct (Hcons w s Hs Hk) as [Hc_ Hnoted].
    exists (node_of_simple s). split; [ exact Hc_ | intros Hused ].
    assert (Hb : existsb (fun n => s_field_used sol n (Field_value_slot w))
                   bound = true).
    { apply existsb_exists. exists a. auto. }
    destruct (eval_value_slots_lookup' _ _ _ _ _ Hvs'
                (lookup_rebuild_value_slots en bound value_slots w s Hs Hb))
      as (x' & Hx' & Hsx').
    exists x'. split; [ exact Hx' | intros Hhu ].
    destruct (rewrite_simple_eval en vars vars' s x Henv Hnoted Hsx)
      as (x'' & Hs'' & _ & Hkept).
    destruct (Hkept (kept_of_hu en s Hhu)) as [_ Hrel].
    rewrite Hsx' in Hs''. injection Hs'' as <-. exact Hrel.
Qed.

(* The bindings of a [Let_static] of the fragment: those of the symbols
   with a use are kept, with their fields used related; the closures of a
   static set are kept if a symbol of the set has a use, with the value
   slots that the rebuild keeps. *)
Lemma static_bindings_rel (en : rebuild_env) (vars vars' : env) :
  env_ok (env_kinds en) vars vars' ->
  ∀ (bindings : list static_binding) (statics statics' statics2 : static_env),
    forallb (binding_checked exact) bindings = true ->
    every binding_edges_ bindings ->
    statics_rel statics statics' ->
    eval_static_bindings vars bindings statics = Some statics2 ->
    ∃ statics2' : static_env,
      eval_static_bindings vars' (flat_map (rebuild_b en) bindings) statics'
      = Some statics2'
      ∧ statics_rel statics2 statics2'.
Proof.
  intros Henv. induction bindings as [ | b bindings IH];
    intros statics statics' statics2 Hfrag Hedges Hst Hev; simpl in Hev.
  - injection Hev as <-. exists statics'. split; [ reflexivity | exact Hst ].
  - destruct Hedges as [Hb Hbindings]. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hfrag_b Hfrag].
    destruct (binding_checked_cases _ _ Hfrag_b)
      as [Hnc | (id & c & -> & _ & _)].
    2: { simpl in Hev.
      destruct (lookup id (static_code statics)) eqn:Hl; [ discriminate Hev | ].
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      assert (Hl' : lookup id (static_code statics') = None).
      { destruct (Hcode id) as [H | H]; [ exact H | ].
        rewrite H, Hl. reflexivity. }
      cbn [flat_map rebuild_binding]. unfold is_code_id_used.
      destruct (s_has_usage sol (Node_code_id id)).
      - cbn [app eval_static_bindings]. rewrite Hl'.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ exact Hnone | split; [ exact Hblocks | split ] ];
          [ exact Honly | ].
        intros id'. simpl.
        destruct (Pos.eqb id' id); [ right; reflexivity | exact (Hcode id') ].
      - cbn [app eval_static_bindings].
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ exact Hnone | split; [ exact Hblocks | split ] ];
          [ exact Honly | ].
        intros id'. simpl.
        destruct (Pos.eqb_spec id' id) as [-> | _];
          [ left; exact Hl' | exact (Hcode id') ]. }
    destruct b as [id c | id | s [ | slot syms decls value_slots | tag fields
                                 | | | ]];
      simpl in Hnc; try discriminate Hnc; simpl in Hb.
    + exact (IH _ _ _ Hfrag Hbindings Hst Hev).
    + apply andb_true_iff in Hnc as [_ Hself].
      destruct (lookup slot syms) as [s0 | ] eqn:Hss; [ | discriminate Hself ].
      apply Pos.eqb_eq in Hself. subst s0.
      destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Hev | ].
      cbn [eval_static_const] in Hev.
      destruct (lookup slot decls) as [id0 | ] eqn:Hid; [ | discriminate Hev ].
      destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
        [ | discriminate Hev ].
      cbv beta iota in Hev.
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      destruct Hb as (_ & _ & _ & Hslots).
      cbn [flat_map rebuild_binding static_const_is_used].
      destruct (existsb (fun '(_, sym) => is_symbol_used sol sym) syms)
        eqn:Hused.
      * set (bound := map (fun '(_, sym) => Node_symbol sym) syms).
        destruct (value_slots_rel en vars vars' bound value_slots vs Henv Hvs)
          as (vs' & Hvs' & Hempty & Hrel).
        assert (Hs_in : Node_symbol s ∈ bound).
        { apply (in_map (fun '(_, sym) => Node_symbol sym) syms (slot, s)).
          exact (lookup_in _ _ _ Hss). }
        cbn [app eval_static_bindings rebuild_static_const].
        rewrite (Hnone s Hl).
        cbn [eval_static_const]. rewrite Hid. fold bound. rewrite Hvs'.
        cbv beta iota.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s);
             [ discriminate Hl' | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right.
           exists slot, syms, decls, vs. split; [ reflexivity | ].
           split; [ | intros _; discriminate ].
           right. exists vs'. split; [ reflexivity | split ].
           ++ intros He. unfold binding_checked in Hfrag_b.
              rewrite He in Hfrag_b. simpl in Hfrag_b.
              destruct value_slots;
                [ exact (Hempty eq_refl) | discriminate Hfrag_b ].
           ++ exact (Hrel _ Hs_in Hslots).
      * cbn [app]. refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl'.
           destruct (Pos.eqb s' s);
             [ discriminate Hl' | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Honly s' allocated Hl') ].
           injection Hl' as <-. right.
           exists slot, syms, decls, vs. split; [ reflexivity | ].
           split; [ left; exact (Hnone s Hl) | intros Hu' ].
           rewrite (lookup_existsb (is_symbol_used sol) slot syms s Hss Hu')
             in Hused.
           discriminate Hused.
    + destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Hev | ].
      cbn [eval_static_const] in Hev.
      destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
        [ | discriminate Hev ].
      cbv beta iota in Hev.
      destruct Hb as (Hcons & _ & _ & Hnoted).
      destruct Hst as (Hnone & Hblocks & Honly & Hcode).
      cbn [flat_map]. rewrite rebuild_binding_block. unfold is_symbol_used.
      destruct (s_has_usage sol (Node_symbol s)) eqn:Hu.
      * destruct (rebuild_fields_eval en vars vars' (Node_symbol s) Henv
                    fields 0 vs Hcons Hnoted Hvs)
          as (vs' & Hvs' & Hf & _).
        cbn [app eval_static_bindings]. rewrite (Hnone s Hl).
        cbn [eval_static_const]. rewrite Hvs'. cbv beta iota.
        refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl' |- *.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ | exact (Hblocks s' tag' fields' Hl' Hu') ].
           injection Hl' as <- <-. exists vs'.
           split; [ reflexivity | exact Hf ].
        -- intros s' allocated Hl'. simpl in Hl' |- *.
           destruct (Pos.eqb s' s);
             [ injection Hl' as <-; left; eexists _, _; reflexivity
             | exact (Honly s' allocated Hl') ].
      * cbn [app]. refine (IH _ _ _ Hfrag Hbindings _ Hev).
        split; [ | split; [ | split; [ | intros id'; exact (Hcode id') ] ] ].
        -- intros s' Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s); [ discriminate Hl' | exact (Hnone s' Hl') ].
        -- intros s' tag' fields' Hl' Hu'. simpl in Hl'.
           destruct (Pos.eqb_spec s' s) as [-> | _];
             [ congruence | exact (Hblocks s' tag' fields' Hl' Hu') ].
        -- intros s' allocated Hl'. simpl in Hl'.
           destruct (Pos.eqb s' s);
             [ injection Hl' as <-; left; eexists _, _; reflexivity
             | exact (Honly s' allocated Hl') ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Parameters                                                               *)
(* ------------------------------------------------------------------------ *)

(* The parameters kept by a mask, bound in the second run to the values
   kept: each parameter that the rebuild keeps is bound to a related
   value. *)
Lemma bind_params_rel (params : list variable) :
  ∀ (vs vs' : list value) (mask : list bool)
    (kinds : finite_map flambda_kind) (vars vars' hv : env),
    env_ok kinds vars vars' ->
    length vs' = length vs ->
    length mask = length params ->
    (∀ (i : nat) (p : variable) (v v' : value),
       nth_error params i = Some p -> nth_error vs i = Some v ->
       nth_error vs' i = Some v' ->
       is_var_used sol p (kind_of_value v) = true ->
       nth_error mask i = Some true ∧ rel v (Node_variable p) v') ->
    bind_params params vs vars = Some hv ->
    ∃ hv' : env,
      bind_params (filter_kept mask params) (filter_kept mask vs') vars'
      = Some hv'
      ∧ env_ok (rev (combine params (map kind_of_value vs)) ++ kinds) hv hv'.
Proof.
  induction params as [ | p params IH];
    intros [ | v vs] [ | v' vs'] [ | m mask] kinds vars vars' hv Henv Hlen
      Hmlen Hrel Hbind;
    simpl in Hbind, Hlen, Hmlen; try discriminate.
  - injection Hbind as <-. exists vars'. split; [ reflexivity | exact Henv ].
  - assert (Hrel' : ∀ (i : nat) (p' : variable) (w w' : value),
              nth_error params i = Some p' -> nth_error vs i = Some w ->
              nth_error vs' i = Some w' ->
              is_var_used sol p' (kind_of_value w) = true ->
              nth_error mask i = Some true ∧ rel w (Node_variable p') w').
    { intros i. exact (Hrel (S i)). }
    assert (Henv' : env_ok ((p, kind_of_value v) :: kinds) ((p, v) :: vars)
                      (if m then (p, v') :: vars' else vars')).
    { apply (env_ok_bind kinds vars vars'); [ exact Henv | | ].
      - intros Hu. destruct (Hrel 0 p v v' eq_refl eq_refl eq_refl Hu)
          as [[= ->] Hr].
        exists v'. auto.
      - destruct m; [ right; exists v'; reflexivity | left; reflexivity ]. }
    destruct (IH vs vs' mask _ _ _ hv Henv' ltac:(lia) ltac:(lia) Hrel' Hbind)
      as (hv' & Hbind' & Hok).
    exists hv'. split.
    + destruct m; exact Hbind'.
    + simpl. rewrite <- app_assoc. exact Hok.
Qed.

Lemma with_kinds_combine (params : list variable) :
  ∀ kinds : list flambda_kind,
    length kinds = length params ->
    with_kinds params kinds = combine params kinds.
Proof.
  induction params as [ | p params IH]; intros [ | k kinds] Hlen;
    simpl in Hlen |- *; try discriminate; [ reflexivity | ].
  f_equal. apply IH. lia.
Qed.

Lemma with_kinds_length (params : list variable) :
  ∀ kinds : list flambda_kind, length (with_kinds params kinds) = length params.
Proof.
  induction params as [ | p params IH]; intros [ | k kinds]; simpl;
    try rewrite IH; reflexivity.
Qed.

Lemma with_kinds_nth (params : list variable) :
  ∀ (kinds : list flambda_kind) (i : nat) (p : variable) (k : flambda_kind),
    nth_error params i = Some p -> nth_error kinds i = Some k ->
    nth_error (with_kinds params kinds) i = Some (p, k).
Proof.
  induction params as [ | p0 params IH]; intros [ | k0 kinds] i p k Hp Hk;
    try (destruct i; discriminate).
  destruct i as [ | i]; simpl in Hp, Hk |- *.
  - injection Hp as <-. injection Hk as <-. reflexivity.
  - exact (IH kinds i p k Hp Hk).
Qed.

Lemma handler_mask_length (b : bool) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  length (handler_mask sol b (Handler params metadata handler))
  = length params.
Proof.
  unfold handler_mask, params_to_keep. rewrite length_map.
  apply with_kinds_length.
Qed.

(* A parameter that the rebuild keeps is kept by the mask. *)
Lemma handler_mask_used (b : bool) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (i : nat) (p : variable)
    (k : flambda_kind) :
  nth_error params i = Some p ->
  nth_error (hm_params_kinds metadata) i = Some k ->
  is_var_used sol p k = true ->
  nth_error (handler_mask sol b (Handler params metadata handler)) i
  = Some true.
Proof.
  intros Hp Hk Hu. unfold handler_mask, params_to_keep.
  rewrite nth_error_map, (with_kinds_nth _ _ _ _ _ Hp Hk). simpl.
  rewrite Hu. reflexivity.
Qed.

Lemma filter_kept_map {A B : Type} (f : A -> B) (mask : list bool) :
  ∀ l : list A, filter_kept mask (map f l) = map f (filter_kept mask l).
Proof.
  induction mask as [ | m mask IH]; intros [ | a l]; simpl; try reflexivity.
  - f_equal. clear a. induction l as [ | a l IHl]; simpl; [ reflexivity | ].
    f_equal. exact IHl.
  - destruct m; simpl; [ f_equal | ]; apply IH.
Qed.

Lemma eval_simples_filter_kept (vars : env) (mask : list bool) :
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs ->
    eval_simples vars (filter_kept mask args) = Some (filter_kept mask vs).
Proof.
  induction mask as [ | m mask IH]; intros args vs Hvs.
  - assert (Hnil : ∀ l : list simple, filter_kept [] l = l).
    { induction l as [ | a l IHl]; simpl;
        [ reflexivity | f_equal; exact IHl ]. }
    assert (Hnil' : ∀ l : list value, filter_kept [] l = l).
    { induction l as [ | a l IHl]; simpl;
        [ reflexivity | f_equal; exact IHl ]. }
    rewrite Hnil, Hnil'. exact Hvs.
  - destruct args as [ | s args]; simpl in Hvs.
    + injection Hvs as <-. reflexivity.
    + destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
      destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
      injection Hvs as <-. simpl. destruct m.
      * simpl. rewrite Hs, (IH args ws Hws). reflexivity.
      * exact (IH args ws Hws).
Qed.

Lemma with_params_kinds_kinds (metadata : handler_metadata)
    (kinds : list flambda_kind) :
  hm_params_kinds (with_params_kinds metadata kinds) = kinds.
Proof.
  unfold with_params_kinds. destruct (hm_is_exn_handler metadata); reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Lookups in a recursive group                                             *)
(* ------------------------------------------------------------------------ *)

Lemma lookup_map_pairs {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun '(k', a) => (k', f a)) l)
  = match lookup k l with Some a => Some (f a) | None => None end.
Proof.
  induction l as [ | [k' a] l IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma lookup_map_pairs_app {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) (rest : list (positive * B)) :
  lookup k (map (fun '(k', a) => (k', f a)) l ++ rest)
  = match lookup k l with Some a => Some (f a) | None => lookup k rest end.
Proof.
  rewrite lookup_app, lookup_map_pairs. destruct (lookup k l); reflexivity.
Qed.

Lemma lookup_rec_closures
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts rest : cont_env) (k : continuation) :
  lookup k (rec_closures handlers vars conts ++ rest)
  = match lookup k handlers with
    | Some h => Some (Closure h vars conts handlers)
    | None => lookup k rest
    end.
Proof.
  unfold rec_closures.
  exact (lookup_map_pairs_app (fun h => Closure h vars conts handlers) k
           handlers rest).
Qed.

Lemma lookup_none_among (k : positive) {A : Type} (l : list (positive * A)) :
  lookup k l = None -> among k (map fst l) = false.
Proof.
  intros H. destruct (among k (map fst l)) eqn:Hin; [ | reflexivity ].
  apply among_In in Hin. destruct (lookup_none_not_in k l H Hin).
Qed.

(* ------------------------------------------------------------------------ *)
(* The rebuild, construct by construct                                      *)
(* ------------------------------------------------------------------------ *)

(* The cases of the rebuild of a [Let]. *)
Lemma rebuild_let_cases (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  rebuild en (Let x kind d body) = Invalid
  ∨ (∃ (d' : named) (en' : rebuild_env),
       rebuild en (Let x kind d body) = Let x kind d' (rebuild en' body))
  ∨ ∃ en' : rebuild_env, rebuild en (Let x kind d body) = rebuild en' body.
Proof.
  cbn [rebuild_expr].
  destruct (is_pending_projection en x d); [ right; left; eauto | ].
  destruct d; cbv zeta;
    repeat match goal with
           | |- context [if ?b then _ else _] => destruct b
           end; eauto.
Qed.

(* A [Let] that is not of a set of closures, with no projection pending,
   or the next projection pending, of a function slot. *)

Lemma rebuild_let_pending_eq (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (move_from move_to : function_slot)
    (closure : simple) (body : expr) (ys : list variable) :
  env_projections en = x :: ys ->
  rebuild en (Let x kind (Project_function_slot move_from move_to closure)
                body)
  = Let x kind (Project_function_slot move_from move_to closure)
      (rebuild (bind_variables_then en [(x, kind)] ys) body).
Proof.
  intros Hep. cbn [rebuild_expr]. unfold is_pending_projection.
  rewrite Hep, Pos.eqb_refl. reflexivity.
Qed.

(* A closure of a set: it is kept if it or a closure projected from it
   next has a use, and these are then pending. *)
Lemma rebuild_let_set_eq (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (body : expr) :
  env_projections en = [] ->
  rebuild en (Let x kind (Closure_of_set slot decls value_slots) body)
  = if is_dead_var sol x kind
       || existsb (fun y => is_dead_var sol y Value)
            (projected_closures x slot (other_slots slot decls) body)
    then Invalid
    else if is_var_used sol x kind
            || existsb (fun y => is_var_used sol y Value)
                 (projected_closures x slot (other_slots slot decls) body)
    then
      Let x kind
        (Closure_of_set slot decls
           (rebuild_value_slots sol en (set_closure_nodes F (Node_variable x))
              value_slots))
        (rebuild
           (bind_variables_then en [(x, kind)]
              (projected_closures x slot (other_slots slot decls) body))
           body)
    else rebuild (bind_variables en [(x, kind)]) body.
Proof.
  intros Hep. destruct en as [ec ek ep]. simpl in Hep. subst ep.
  reflexivity.
Qed.

Lemma rebuild_let_eq (en : rebuild_env) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  env_projections en = [] -> named_plain d = true ->
  rebuild en (Let x kind d body)
  = if is_dead_var sol x kind then Invalid
    else if is_var_used sol x kind
    then
      Let x kind (rebuild_named sol en x d)
        (rebuild (bind_variables en [(x, kind)]) body)
    else rebuild (bind_variables en [(x, kind)]) body.
Proof.
  intros Hep Hd. destruct en as [ec ek ep]. simpl in Hep. subst ep.
  destruct d; try discriminate Hd; reflexivity.
Qed.

Definition rec_env (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) : rebuild_env :=
  bind_continuations en
    (map (fun '(k, h) => (k, handler_mask sol false h)) handlers).

Definition rebuilt_handlers (en : rebuild_env)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k, h) => (k, rebuild_h en false (handler_mask sol false h) h))
    handlers.

Lemma rebuild_let_cont_recursive_eq (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  rebuild en (Let_cont_recursive handlers body)
  = Let_cont_recursive (rebuilt_handlers (rec_env en handlers) handlers)
      (rebuild (rec_env en handlers) body).
Proof. reflexivity. Qed.

Lemma map_fst_rebuilt_handlers (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) :
  map fst (rebuilt_handlers en handlers) = map fst handlers.
Proof.
  unfold rebuilt_handlers. rewrite map_map.
  apply map_ext. intros [k h]. reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Terms of the fragment                                                    *)
(* ------------------------------------------------------------------------ *)

(* The rebuild names no continuation that the term does not. *)
Lemma free_rebuild :
  (∀ e : expr,
     checked exact e = true ->
     ∀ (en : rebuild_env) (k : continuation),
       k ∈ free_continuations_of_expr (rebuild en e) ->
       k ∈ free_continuations_of_expr e)
  ∧ (∀ h : continuation_handler,
       handler_checked exact h = true ->
       ∀ (en : rebuild_env) (dead_check : bool) (mask : list bool)
         (k : continuation),
         k ∈ free_continuations_of_handler (rebuild_h en dead_check mask h) ->
         k ∈ free_continuations_of_handler h).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH Hfrag en k. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [_ Hbody].
    destruct (rebuild_let_cases en x kind d body)
      as [-> | [(d' & en' & ->) | (en' & ->)]]; [ intros [] | | ];
      simpl; apply IH; exact Hbody.
  - intros bindings body Hcodes IH Hfrag en k. simpl in Hfrag |- *.
    apply andb_true_iff in Hfrag as [Hbindings Hbody].
    intros Hin. apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as (b & Hb & Hk).
      apply in_flat_map in Hb as (b0 & Hb0 & Hb).
      rewrite forallb_forall in Hbindings. specialize (Hbindings b0 Hb0).
      destruct b0 as [id c | id | s const].
      * cbn [rebuild_binding] in Hb.
        match type of Hb with
        | context [if ?b then _ else _] => destruct b
        end; destruct Hb as [<- | []]; [ | destruct Hk ].
        apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hb0 | ].
        destruct (binding_checked_cases _ _ Hbindings)
          as [Hnc0 | (id' & c' & Heq & He & Hnc0)]; [ discriminate Hnc0 | ].
        injection Heq as <- <-.
        assert (Hcb : checked exact (code_body c) = true)
          by (rewrite He; exact (no_code_checked _ Hnc0)).
        pose proof (Hcodes id c Hb0 Hcb) as IHc.
        destruct c as [params mc ret exn md cbody]. simpl in Hk |- *.
        unfold without in Hk |- *. apply filter_In in Hk as [Hk Hnot].
        apply filter_In. split; [ | exact Hnot ].
        match type of Hk with
        | context [if ?b then _ else _] => destruct b
        end; [ destruct Hk | exact (IHc _ _ Hk) ].
      * simpl in Hb. destruct Hb as [<- | []]. destruct Hk.
      * cbn [rebuild_binding] in Hb.
        match type of Hb with
        | context [if ?b then _ else _] => destruct b
        end; [ | destruct Hb ].
        destruct Hb as [<- | []]. destruct Hk.
    + apply in_or_app. right. exact (IH Hbody _ _ Hin).
  - intros k0 h body IHh IH Hfrag en k. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hh Hbody]. cbn [rebuild_expr].
    set (mask := handler_mask sol (handler_is_exn_handler h) h).
    set (body' := rebuild (bind_continuations en [(k0, mask)]) body).
    simpl. intros Hin. apply in_or_app.
    destruct (among k0 (free_continuations_of_expr body')) eqn:Hk0.
    + simpl in Hin. apply in_app_or in Hin as [Hin | Hin].
      * left. exact (IHh Hh _ _ _ _ Hin).
      * right. apply among_In. apply among_In in Hin.
        rewrite among_without in Hin |- *.
        apply andb_true_iff in Hin as [Hin Hnot]. rewrite Hnot, andb_true_r.
        apply among_In. exact (IH Hbody _ _ (proj1 (among_In _ _) Hin)).
    + right. apply among_In. rewrite among_without.
      apply andb_true_intro. split.
      * apply among_In. exact (IH Hbody _ _ Hin).
      * simpl. destruct (Pos.eqb_spec k k0) as [-> | _]; [ | reflexivity ].
        apply among_In in Hin. rewrite Hin in Hk0. discriminate.
  - intros handlers body IHh IH Hfrag en k. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    rewrite rebuild_let_cont_recursive_eq. simpl.
    rewrite map_fst_rebuilt_handlers. intros Hin.
    apply among_In. apply among_In in Hin.
    rewrite among_without in Hin |- *.
    apply andb_true_iff in Hin as [Hin Hnot]. rewrite Hnot, andb_true_r.
    apply among_In. apply among_In in Hin.
    apply in_app_or in Hin as [Hin | Hin]; apply in_or_app; [ left | right ].
    + apply in_flat_map in Hin as ([k' h'] & Hkh & Hk).
      unfold rebuilt_handlers in Hkh. apply in_map_iff in Hkh
        as ([k'' h''] & [= <- <-] & Hin').
      apply in_flat_map. exists (k'', h''). split; [ exact Hin' | ].
      exact (IHh k'' h'' Hin' (Hhandlers _ Hin') _ _ _ _ Hk).
    + exact (IH Hbody _ _ Hin).
  - intros ap Hfrag. discriminate Hfrag.
  - intros ac _ en k. cbn [rebuild_expr]. unfold rebuild_apply_cont.
    destruct (rewrite_apply_cont sol en ac) as [ac' | ] eqn:Hrw;
      [ | intros [] ].
    unfold rewrite_apply_cont in Hrw.
    destruct (existsb (simple_has_no_source sol) (ac_args ac));
      [ discriminate Hrw | ].
    injection Hrw as <-. simpl. exact (fun H => H).
  - intros scrutinee arms _ en k. cbn [rebuild_expr]. unfold rebuild_switch.
    cbv zeta.
    destruct (Keyed.is_empty _); [ intros [] | ].
    simpl. intros Hin. apply in_flat_map in Hin as ([d ac'] & Hin & Hk).
    apply in_flat_map in Hin as ([d' ac] & Hcomp & Hin).
    destruct (rewrite_apply_cont sol en ac) as [ac'' | ] eqn:Hrw;
      [ | destruct Hin ].
    destruct Hin as [[= <- <-] | []].
    apply in_flat_map. exists (d', ac). split.
    + apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hcomp.
      exact (keyed_lookup_in Z.eqb Z.eqb_eq d' ac arms Hcomp).
    + unfold rewrite_apply_cont in Hrw.
      destruct (existsb (simple_has_no_source sol) (ac_args ac));
        [ discriminate Hrw | ].
      injection Hrw as <-. exact Hk.
  - intros _ en k [].
  - intros params metadata handler IH Hfrag en dead_check mask k.
    simpl. destruct (dead_check && _); [ intros [] | ].
    apply IH. exact Hfrag.
Qed.

(* The rebuild binds nothing static that the term does not. *)
Lemma binds_statics_rebuild :
  (∀ e : expr,
     checked exact e = true ->
     ∀ en : rebuild_env,
       binds_statics (rebuild en e) = true -> binds_statics e = true)
  ∧ (∀ h : continuation_handler,
       handler_checked exact h = true ->
       ∀ (en : rebuild_env) (dead_check : bool) (mask : list bool),
         handler_binds_statics (rebuild_h en dead_check mask h) = true ->
         handler_binds_statics h = true).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH Hfrag en. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [_ Hbody].
    destruct (rebuild_let_cases en x kind d body)
      as [-> | [(d' & en' & ->) | (en' & ->)]]; [ discriminate | | ];
      simpl; apply IH; exact Hbody.
  - intros bindings body _ IH Hfrag en. simpl in Hfrag |- *.
    apply andb_true_iff in Hfrag as [Hbindings Hbody].
    intros Hb. apply orb_true_iff in Hb as [Hb | Hb].
    + apply orb_true_intro. left.
      apply existsb_exists in Hb as (b & Hin & Hbinds).
      apply in_flat_map in Hin as (b0 & Hb0 & Hin).
      apply existsb_exists. exists b0. split; [ exact Hb0 | ].
      rewrite forallb_forall in Hbindings. specialize (Hbindings b0 Hb0).
      destruct b0 as [id c | id | s const]; [ reflexivity | | reflexivity ].
      simpl in Hin. destruct Hin as [<- | []]. discriminate Hbinds.
    + apply orb_true_intro. right. exact (IH Hbody _ Hb).
  - intros k0 h body IHh IH Hfrag en. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hh Hbody]. cbn [rebuild_expr].
    destruct (among k0 _); simpl; intros Hb;
      [ apply orb_true_iff in Hb as [Hb | Hb]; apply orb_true_intro;
        [ left; exact (IHh Hh _ _ _ Hb) | right; exact (IH Hbody _ Hb) ]
      | apply orb_true_intro; right; exact (IH Hbody _ Hb) ].
  - intros handlers body IHh IH Hfrag en. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    rewrite rebuild_let_cont_recursive_eq. simpl. intros Hb.
    apply orb_true_iff in Hb as [Hb | Hb]; apply orb_true_intro;
      [ left | right; exact (IH Hbody _ Hb) ].
    apply existsb_exists in Hb as ([k h'] & Hin & Hbinds).
    unfold rebuilt_handlers in Hin.
    apply in_map_iff in Hin as ([k' h''] & [= <- <-] & Hin).
    apply existsb_exists. exists (k', h''). split; [ exact Hin | ].
    exact (IHh k' h'' Hin (Hhandlers _ Hin) _ _ _ Hbinds).
  - intros ap Hfrag. discriminate Hfrag.
  - intros ac _ en. cbn [rebuild_expr]. unfold rebuild_apply_cont.
    destruct (rewrite_apply_cont sol en ac); discriminate.
  - intros scrutinee arms _ en. cbn [rebuild_expr]. unfold rebuild_switch.
    cbv zeta. destruct (Keyed.is_empty _); discriminate.
  - intros _ en. discriminate.
  - intros params metadata handler IH Hfrag en dead_check mask.
    simpl. destruct (dead_check && _); [ discriminate | ].
    apply IH. exact Hfrag.
Qed.

Lemma rebuild_let_static_eq (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  rebuild en (Let_static bindings body)
  = Let_static (flat_map (rebuild_b en) bindings)
      (rebuild (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma rebuild_let_cont_non_recursive_eq (en : rebuild_env)
    (k : continuation) (h : continuation_handler) (body : expr) :
  rebuild en (Let_cont_non_recursive k h body)
  = if among k
         (free_continuations_of_expr
            (rebuild
               (bind_continuations en
                  [(k, handler_mask sol (handler_is_exn_handler h) h)])
               body))
    then
      Let_cont_non_recursive k
        (rebuild_h en true (handler_mask sol (handler_is_exn_handler h) h) h)
        (rebuild
           (bind_continuations en
              [(k, handler_mask sol (handler_is_exn_handler h) h)])
           body)
    else
      rebuild
        (bind_continuations en
           [(k, handler_mask sol (handler_is_exn_handler h) h)])
        body.
Proof. reflexivity. Qed.

Lemma rebuild_handler_eq (en : rebuild_env) (dead_check : bool)
    (mask : list bool) (params : list variable) (metadata : handler_metadata)
    (handler : expr) :
  rebuild_h en dead_check mask (Handler params metadata handler)
  = Handler (filter_kept mask params)
      (with_params_kinds metadata
         (filter_kept mask (hm_params_kinds metadata)))
      (if dead_check
          && existsb (fun '(x, kind) => is_dead_var sol x kind)
               (with_kinds params (hm_params_kinds metadata))
       then Invalid
       else
         rebuild
           (bind_variables en (with_kinds params (hm_params_kinds metadata)))
           handler).
Proof. reflexivity. Qed.

(* The [Apply_cont] that the rebuild makes of one whose arguments have
   sources. *)
Definition rebuilt_apply_cont (en : rebuild_env) (ac : apply_cont)
    : apply_cont :=
  {| ac_continuation := ac_continuation ac;
     ac_args :=
       match lookup (ac_continuation ac) (env_conts en) with
       | Some mask =>
         filter_kept mask (map (rewrite_simple sol en) (ac_args ac))
       | None => map (rewrite_simple sol en) (ac_args ac)
       end;
     ac_trap_action := ac_trap_action ac |}.

Lemma rewrite_apply_cont_some (en : rebuild_env) (ac : apply_cont) :
  existsb (simple_has_no_source sol) (ac_args ac) = false ->
  rewrite_apply_cont sol en ac = Some (rebuilt_apply_cont en ac).
Proof. intros H. unfold rewrite_apply_cont. rewrite H. reflexivity. Qed.

Lemma eval_simples_member (vars : env) (args : list simple)
    (vs : list value) (s : simple) :
  eval_simples vars args = Some vs -> s ∈ args ->
  ∃ w : value, eval_simple vars s = Some w.
Proof.
  revert vs. induction args as [ | s' args IH]; intros vs Hvs Hin;
    [ destruct Hin | ]. simpl in Hvs.
  destruct (eval_simple vars s') as [w | ] eqn:Hs; [ | discriminate ].
  destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
  destruct Hin as [<- | Hin];
    [ exists w; exact Hs | exact (IH ws eq_refl Hin) ].
Qed.

(* The arguments of an [Apply_cont] that the first run evaluates have
   sources. *)
Lemma args_have_sources (vars : env) (args : list simple) (vs : list value) :
  env_flows_ vars -> simples_noted noted args ->
  eval_simples vars args = Some vs ->
  existsb (simple_has_no_source sol) args = false.
Proof.
  intros Hvars Hnoted Hvs. apply not_true_iff_false. intros Hex.
  apply existsb_exists in Hex as (s & Hin & Hno).
  destruct (eval_simples_member vars args vs s Hvs Hin) as (w & Hw).
  pose proof (flows_hs is_local G sol keep called Hc _ _
                (simple_flows is_local G sol noted cds keep Hc Hconsts Hsyms _ _
                   _ Hvars (Hnoted s Hin) Hw)) as Hhs.
  destruct s; simpl in Hno, Hhs; try discriminate Hno;
    rewrite Hhs in Hno; discriminate Hno.
Qed.

Lemma bind_params_length (params : list variable) :
  ∀ (vs : list value) (vars hv : env),
    bind_params params vs vars = Some hv -> length vs = length params.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars hv H;
    simpl in H; try discriminate; [ reflexivity | ].
  simpl. f_equal. exact (IH vs _ _ H).
Qed.

Lemma with_kinds_in (params : list variable) :
  ∀ (kinds : list flambda_kind) (x : variable) (k : flambda_kind),
    (x, k) ∈ with_kinds params kinds -> x ∈ params.
Proof.
  induction params as [ | p params IH]; intros [ | k0 kinds] x k Hin;
    simpl in Hin; try contradiction;
    (destruct Hin as [[= -> _] | Hin]; [ left; reflexivity | right ];
     exact (IH _ x k Hin)).
Qed.

(* The parameters of a handler entered have sources: its rebuild is not
   [Invalid]. *)
Lemma params_not_dead (params : list variable) (kinds : list flambda_kind)
    (vs : list value) (vars hv : env) :
  bind_params params vs vars = Some hv -> env_flows_ hv ->
  existsb (fun '(x, kind) => is_dead_var sol x kind)
    (with_kinds params kinds) = false.
Proof.
  intros Hbind Hhv. apply not_true_iff_false. intros Hex.
  apply existsb_exists in Hex as ([x kind] & Hin & Hdead).
  destruct (bind_params_bound params vs vars hv x Hbind
              (or_introl (with_kinds_in _ _ _ _ Hin)))
    as (w & Hw).
  pose proof (flows_hs is_local G sol keep called Hc _ _ (Hhv x w Hw)) as Hhs.
  unfold is_dead_var in Hdead. rewrite Hhs in Hdead.
  destruct kind; discriminate Hdead.
Qed.

(* Entering a handler in the two runs: the second, at the parameters and
   values kept, under an environment related to the first's. *)
Lemma enter_handler (b : bool) (params : list variable)
    (metadata : handler_metadata) (handler body' : expr) (en0 : rebuild_env)
    (vars0 vars0' hv : env) (conts0' : cont_env)
    (rec' : list (continuation * continuation_handler))
    (vs vs' : list value) (statics' : static_env) (hp' : heap) :
  env_ok (env_kinds en0) vars0 vars0' ->
  map kind_of_value vs = hm_params_kinds metadata ->
  map kind_of_value vs' = map kind_of_value vs ->
  bind_params params vs vars0 = Some hv ->
  (∀ (i : nat) (p : variable) (v v' : value),
     nth_error params i = Some p -> nth_error vs i = Some v ->
     nth_error vs' i = Some v' ->
     is_var_used sol p (kind_of_value v) = true ->
     rel v (Node_variable p) v') ->
  ∃ hv' : env,
    apply_cont_value
      (Closure
         (Handler
            (filter_kept (handler_mask sol b (Handler params metadata handler))
               params)
            (with_params_kinds metadata
               (filter_kept
                  (handler_mask sol b (Handler params metadata handler))
                  (hm_params_kinds metadata)))
            body')
         vars0' conts0' rec')
      (filter_kept (handler_mask sol b (Handler params metadata handler)) vs')
      statics' hp'
    = Step (State body' hv' (rec_closures rec' vars0' conts0' ++ conts0')
              statics' hp')
    ∧ env_ok
        (env_kinds
           (bind_variables en0 (with_kinds params (hm_params_kinds metadata))))
        hv hv'.
Proof.
  intros Henv0 Hkinds Hkinds' Hbind Hrel.
  set (mask := handler_mask sol b (Handler params metadata handler)).
  pose proof (bind_params_length params vs vars0 hv Hbind) as Hlen.
  assert (Hlen' : length vs' = length vs).
  { rewrite <- (length_map kind_of_value vs'), Hkinds', length_map.
    reflexivity. }
  destruct (bind_params_rel params vs vs' mask (env_kinds en0) vars0 vars0' hv
              Henv0 Hlen' (handler_mask_length b params metadata handler))
    as (hv' & Hbind' & Hok); [ | exact Hbind | ].
  { intros i p v v' Hp Hv Hv' Hu.
    split; [ | exact (Hrel i p v v' Hp Hv Hv' Hu) ].
    apply (handler_mask_used b params metadata handler i p (kind_of_value v)
             Hp);
      [ | exact Hu ].
    rewrite <- Hkinds, nth_error_map, Hv. reflexivity. }
  exists hv'. split.
  - cbn [apply_cont_value]. rewrite with_params_kinds_kinds.
    rewrite <- filter_kept_map, Hkinds', Hkinds, kinds_eqb_refl.
    rewrite Hbind'. reflexivity.
  - cbn [env_kinds bind_variables bind_variables_then].
    rewrite (with_kinds_combine params (hm_params_kinds metadata)),
      <- Hkinds; [ exact Hok | ].
    rewrite <- Hkinds, length_map. exact Hlen.
Qed.

(* The arguments of an [Apply_cont] of a handler, at the parameters
   that the rebuild keeps. *)
Lemma apply_cont_args_rel (en : rebuild_env) (vars vars' : env)
    (ce : finite_map (list node)) (ac : apply_cont) (params : list variable)
    (vs vs' : list value) :
  env_ok (env_kinds en) vars vars' ->
  apply_cont_edges G noted ce ac ->
  lookup (ac_continuation ac) ce = Some (map Node_variable params) ->
  eval_simples vars (ac_args ac) = Some vs ->
  (∀ (i : nat) (s : simple) (v v' : value),
     nth_error (ac_args ac) i = Some s -> nth_error vs i = Some v ->
     nth_error vs' i = Some v' -> kept_simple en s = true ->
     rel v (node_of_simple s) v') ->
  ∀ (i : nat) (p : variable) (v v' : value),
    nth_error params i = Some p -> nth_error vs i = Some v ->
    nth_error vs' i = Some v' ->
    is_var_used sol p (kind_of_value v) = true ->
    rel v (Node_variable p) v'.
Proof.
  intros Henv [Haliases _] Hce Hvs Hrels i p v v' Hp Hv Hv' Hu.
  destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
  assert (Hal : s_alias sol (Node_variable p) (node_of_simple s) = true).
  { apply (alias_input _ _ _ Hc). apply (Haliases _ Hce).
    apply (nth_error_combine _ _ i); [ | exact Hs ].
    rewrite nth_error_map, Hp. reflexivity. }
  apply (rel_alias G sol keep (exact = true) is_local Hc v _ _ v' Hal).
  apply (Hrels i s v v' Hs Hv Hv').
  destruct (is_var_used_cases p _ Hu) as [Hhu | Hk].
  - apply kept_of_hu. exact (has_usage_alias _ _ _ Hc _ _ Hhu Hal).
  - exact (kept_of_region en vars vars' s v Henv Hsv Hk).
Qed.

(* ------------------------------------------------------------------------ *)
(* The relation                                                             *)
(* ------------------------------------------------------------------------ *)

Definition unit_cont (k : continuation) : Prop :=
  k = unit_return u ∨ k = unit_exn u.

(* The nodes of the parameters of a handler. *)
Definition cont_params (c : cont_value) : list node :=
  match c with
  | Closure h _ _ _ => handler_params h
  | _ => []
  end.

(* The continuations [ks] in the two runs, with the masks of the rebuild
   and the nodes of the parameters ([ce]): ways out of the term, through
   a continuation of the unit, or continuations related by [R]. *)
Definition conts_related (R : list bool -> cont_value -> cont_value -> Prop)
    (ks : list continuation) (conts conts' : cont_env)
    (masks : finite_map (list bool)) (ce : finite_map (list node)) : Prop :=
  ∀ k : continuation,
    k ∈ ks ->
    (lookup k conts = None ∧ lookup k conts' = None ∧ lookup k masks = None
     ∧ unit_cont k ∧ lookup k ce = lookup k (unit_conts u))
    ∨ ∃ (mask : list bool) (c c' : cont_value),
        lookup k masks = Some mask ∧ lookup k conts = Some c
        ∧ lookup k conts' = Some c' ∧ lookup k ce = Some (cont_params c)
        ∧ R mask c c'.

(* What the unit asks of an expression. *)
Definition expr_ok (e : expr) : Prop :=
  exits_one_value u e = true ∧ scrutinees_used G e ∧ checked exact e = true.

Definition handler_ok (h : continuation_handler) : Prop :=
  handler_exits_one_value u h = true ∧ handler_scrutinees_used G h
  ∧ handler_checked exact h = true.

(* A handler of the first run, and its rebuild in the second, with the
   mask of the parameters kept; the facts of the handler are in the
   graph, under the nodes of the continuations of its definition. *)
#[local] Set Warnings "-register-all".
Inductive cont_rel : list bool -> cont_value -> cont_value -> Prop :=
  | Cont_rel_non_recursive (h : continuation_handler) (en : rebuild_env)
      (vars vars' : env) (conts conts' : cont_env)
      (ce : finite_map (list node)) :
    env_ok (env_kinds en) vars vars' ->
    handler_ok h ->
    handler_edges_ ce h ->
    conts_related cont_rel
      (free_continuations_of_handler
         (rebuild_h en true (handler_mask sol (handler_is_exn_handler h) h)
            h))
      conts conts' (env_conts en) ce ->
    cont_rel (handler_mask sol (handler_is_exn_handler h) h)
      (Closure h vars conts [])
      (Closure
         (rebuild_h en true (handler_mask sol (handler_is_exn_handler h) h)
            h)
         vars' conts' [])
  | Cont_rel_recursive (k : continuation) (h : continuation_handler)
      (en : rebuild_env) (vars vars' : env) (conts conts' : cont_env)
      (handlers : list (continuation * continuation_handler))
      (ce : finite_map (list node)) :
    env_ok (env_kinds en) vars vars' ->
    (k, h) ∈ handlers ->
    (∀ (k' : continuation) (h' : continuation_handler),
       (k', h') ∈ handlers ->
       handler_ok h'
       ∧ handler_edges_ (rec_conts ce handlers) h'
       ∧ conts_related cont_rel
           (without (map fst handlers)
              (free_continuations_of_handler
                 (rebuild_h (rec_env en handlers) false
                    (handler_mask sol false h') h')))
           conts conts' (env_conts en) ce) ->
    cont_rel (handler_mask sol false h)
      (Closure h vars conts handlers)
      (Closure
         (rebuild_h (rec_env en handlers) false (handler_mask sol false h) h)
         vars' conts' (rebuilt_handlers (rec_env en handlers) handlers)).

(* The projections that the rebuild has pending are those of the closures
   of a set that begin the expression, from a variable bound to the same
   closure in the two runs. *)
Definition pending_ok (en : rebuild_env) (vars vars' : env) (e : expr)
    : Prop :=
  env_projections en = []
  ∨ ∃ (x : variable) (slot : function_slot) (others : list function_slot)
      (closure closure' : value),
      env_projections en = projected_closures x slot others e
      ∧ lookup x vars = Some closure ∧ lookup x vars' = Some closure'
      ∧ rel closure (Node_variable x) closure'
      ∧ ∃ (slot0 : function_slot) (decls : list (function_slot * code_id))
          (vs : list (value_slot * value)),
          closure = Value_closure_of_set slot0 decls vs.

(* The two runs: the first has the invariant of the flow, and the facts
   of its expression under the nodes [ce]; the expression of the second
   is the rebuild of the first's; the variables that the rebuild keeps,
   the continuations named and the static environments are related; the
   heaps are the same; and the first expression is as the unit asks. *)
Definition related (s t : state) : Prop :=
  flow_invariant is_local G sol noted deps cds keep codes symbols u s
  ∧ ∃ (en : rebuild_env) (ce : finite_map (list node)),
      edges ce (state_expr s)
      ∧ state_expr t = rebuild en (state_expr s)
      ∧ env_ok (env_kinds en) (state_vars s) (state_vars t)
      ∧ conts_related cont_rel (free_continuations_of_expr (state_expr t))
          (state_conts s) (state_conts t) (env_conts en) ce
      ∧ statics_rel (state_statics s) (state_statics t)
      ∧ state_heap t = state_heap s
      ∧ expr_ok (state_expr s)
      ∧ pending_ok en (state_vars s) (state_vars t) (state_expr s).

Lemma conts_related_sub (ks ks' : list continuation) (conts conts' : cont_env)
    (masks : finite_map (list bool)) (ce : finite_map (list node)) :
  conts_related cont_rel ks conts conts' masks ce ->
  (∀ k : continuation, k ∈ ks' -> k ∈ ks) ->
  conts_related cont_rel ks' conts conts' masks ce.
Proof. intros H Hsub k Hk. exact (H k (Hsub k Hk)). Qed.

(* A continuation bound in front of the maps of the first run, and not
   named. *)
Lemma conts_related_skip (ks : list continuation) (k : continuation)
    (c : cont_value) (mask : list bool) (ns : list node)
    (conts conts' : cont_env) (masks : finite_map (list bool))
    (ce : finite_map (list node)) :
  conts_related cont_rel ks conts conts' masks ce ->
  ~ k ∈ ks ->
  conts_related cont_rel ks ((k, c) :: conts) conts' ((k, mask) :: masks)
    ((k, ns) :: ce).
Proof.
  intros H Hk j Hj. simpl.
  destruct (Pos.eqb_spec j k) as [-> | _]; [ contradiction | exact (H j Hj) ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Steps                                                                    *)
(* ------------------------------------------------------------------------ *)

(* The values of the exits of the two runs. *)
Definition exits_related (vs vs' : list value) : Prop :=
  values_related (fun v v' => observed_equal v v' ∧ (exact = true -> v' = v))
    vs vs'.

Lemma exits_related_eq (vs vs' : list value) :
  exact = true -> exits_related vs vs' -> vs' = vs.
Proof.
  intros He. revert vs'. induction vs as [ | v vs IH]; intros [ | v' vs'] H;
    simpl in H; try contradiction; [ reflexivity | ].
  destruct H as [[_ Hv] H]. rewrite (Hv He), (IH vs' H). reflexivity.
Qed.

Lemma exits_related_observed (vs vs' : list value) :
  exits_related vs vs' -> values_related observed_equal vs vs'.
Proof.
  revert vs'. induction vs as [ | v vs IH]; intros [ | v' vs'] H;
    simpl in H; try contradiction; [ exact I | ].
  destruct H as [[Ho _] H]. split; [ exact Ho | exact (IH vs' H) ].
Qed.

(* Each construct: a step of the first run is answered by the second. *)

Ltac related_intro :=
  let Hflow := fresh "Hflow" in
  intros [Hflow (en & ce & He & Ht & Henv & Hconts & Hst & Hheap & Hok
                 & Hpend)];
  pose proof (step_flow_invariant is_local G sol noted deps cds keep codes
                symbols u Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps Hnd Hcds _
                Hflow) as Hflow';
  pose proof (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ Hflow)
    as (Hvars & _ & Hsflow);
  simpl_state; simpl in He.

(* The variables projected next from a set, in the first run. *)
Lemma pending_let (en : rebuild_env) (vars vars' : env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  pending_ok en vars vars' (Let x kind d body) ->
  env_projections en = []
  ∨ ∃ (x0 : variable) (slot o : function_slot)
      (others : list function_slot) (slot0 : function_slot)
      (decls : list (function_slot * code_id))
      (vs : list (value_slot * value)) (closure' : value),
      d = Project_function_slot slot o (Var x0) ∧ x <> x0
      ∧ env_projections en = x :: projected_closures x0 slot others body
      ∧ lookup x0 vars = Some (Value_closure_of_set slot0 decls vs)
      ∧ lookup x0 vars' = Some closure'
      ∧ rel (Value_closure_of_set slot0 decls vs) (Node_variable x0)
          closure'.
Proof.
  intros [Hep | (x0 & slot & others & closure & closure' & Hep & Hl & Hl'
                 & Hrel & slot0 & decls & vs & ->)];
    [ left; exact Hep | ].
  destruct others as [ | o others]; [ left; exact Hep | ].
  destruct d as [ | | | | | | mf mt [x1 | | ]]; try (left; exact Hep).
  simpl in Hep.
  destruct (Pos.eqb_spec x1 x0) as [-> | _]; [ | left; exact Hep ].
  destruct (Pos.eqb_spec mf slot) as [-> | _]; [ | left; exact Hep ].
  destruct (Pos.eqb_spec mt o) as [-> | _]; [ | left; exact Hep ].
  destruct (Pos.eqb_spec x x0) as [-> | Hne]; [ left; exact Hep | ].
  right. exists x0, slot, o, others, slot0, decls, vs, closure'.
  split; [ reflexivity | split; [ exact Hne | ] ].
  split; [ exact Hep | ].
  split; [ exact Hl | split; [ exact Hl' | exact Hrel ] ].
Qed.

(* No projection is pending before an expression that is not a [Let]. *)
Lemma pending_not_let (en : rebuild_env) (vars vars' : env) (e : expr) :
  pending_ok en vars vars' e ->
  (∀ (x : variable) (slot : function_slot) (others : list function_slot),
     projected_closures x slot others e = []) ->
  env_projections en = [].
Proof.
  intros [Hep | (x & slot & others & _ & _ & Hep & _)] Hnil; [ exact Hep | ].
  rewrite Hep. apply Hnil.
Qed.

(* The function slots of a set other than the first are among its
   slots. *)
Lemma other_slots_bound (slot : function_slot)
    (decls : list (function_slot * code_id)) (o : function_slot) :
  o ∈ other_slots slot decls -> ∃ id : code_id, lookup o decls = Some id.
Proof.
  destruct decls as [ | [first id] others]; simpl; [ intros [] | ].
  destruct (Pos.eqb first slot); [ | intros [] ].
  intros Hin. destruct (Pos.eqb o first); [ eauto | ].
  revert Hin.
  induction others as [ | [g id'] others IH]; simpl; [ intros [] | ].
  intros [-> | Hin]; [ rewrite Pos.eqb_refl; eauto | ].
  destruct (Pos.eqb o g); [ eauto | exact (IH Hin) ].
Qed.

(* The closures projected from a set next have a source: they are
   projected from a closure that flows to the variable of the set. *)
Lemma projections_live (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id)) (vs : list (value_slot * value))
    (id0 : code_id) :
  lookup slot decls = Some id0 ->
  flows_ (Value_closure_of_set slot decls vs) (Node_variable x) ->
  ∀ (others : list function_slot) (ce : finite_map (list node))
    (body : expr),
    (∀ o : function_slot,
       o ∈ others -> ∃ id : code_id, lookup o decls = Some id) ->
    edges ce body ->
    existsb (fun y => is_dead_var sol y Value)
      (projected_closures x slot others body)
    = false.
Proof.
  intros Hslot (sites & Hn & Hall).
  induction others as [ | o others IH]; intros ce body Hothers Hedges;
    [ destruct body; reflexivity | ].
  destruct body as [y k d body | | | | | | | ]; try reflexivity.
  destruct d as [ | | | | | | mf mt [x1 | | ]]; try reflexivity.
  cbn [projected_closures].
  destruct (Pos.eqb_spec x1 x) as [-> | _]; [ | reflexivity ].
  destruct (Pos.eqb_spec mf slot) as [-> | _]; [ | reflexivity ].
  destruct (Pos.eqb_spec mt o) as [-> | _]; [ | reflexivity ].
  destruct (Pos.eqb y x); [ reflexivity | ].
  cbn [andb negb existsb].
  destruct Hedges as [((Hacc & _) & Hslots) Hbody].
  destruct (Hothers o (or_introl eq_refl)) as (id1 & Hl1).
  assert (Hfy : flows_ (Value_closure_of_set o decls vs) (Node_variable y)).
  { destruct (Hall slot id0 Hslot) as (_ & _ & Hfs & _).
    apply (flows_read is_local G sol keep called Hc _ _ _ _ _ _ (Hslots _)
             (Hfs o id1 Hl1) Hacc Hn).
    simpl. exists sites.
    split; [ exact (proj1 (Hall o id1 Hl1)) | exact Hall ]. }
  unfold is_dead_var.
  rewrite (flows_hs is_local G sol keep called Hc _ _ Hfy). simpl.
  exact (IH ce body (fun o' Hin => Hothers o' (or_intror Hin)) Hbody).
Qed.

Lemma let_answered (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (e' : expr) :
  related (State (Let x kind d body) vars conts statics hp)
    (State e' vars' conts' statics' hp') ->
  answered related exits_related measure
    (State (Let x kind d body) vars conts statics hp)
    (State e' vars' conts' statics' hp').
Proof.
  related_intro. destruct He as [Hd Hbody_edges].
  destruct Hok as (Hexits & Hscr & Hcf). simpl in Hexits, Hscr, Hcf.
  apply andb_true_iff in Hcf as [Hfrag Hcf_body].
  unfold answered. rewrite step_let in Hflow' |- *.
  destruct (eval_named statics vars d) as [v | ] eqn:Hev; [ | exact I ].
  destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | exact I ].
  apply kind_eqb_eq in Hkind. subst kind.
  assert (Hflows : flows_ v (Node_variable x)).
  { destruct (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ Hflow')
      as [Hvars' _].
    apply Hvars'. simpl. rewrite Pos.eqb_refl. reflexivity. }
  assert (Hnotdead : is_dead_var sol x (kind_of_value v) = false).
  { pose proof (flows_hs is_local G sol keep called Hc _ _ Hflows) as Hhs.
    unfold is_dead_var. rewrite Hhs.
    destruct (kind_of_value v); reflexivity. }
  destruct (pending_let en vars vars' x (kind_of_value v) d body Hpend)
    as [Hep | (x0 & slot & o & others & slot0 & decls0 & vs0 & c' & -> & Hne
               & Hep & Hx0 & Hx0' & Hrel0)].
  2: { (* The next projection pending: kept as it is. *)
    cbn [eval_named eval_simple] in Hev. rewrite Hx0 in Hev.
    unfold project_function_slot in Hev. simpl in Hev.
    destruct (Pos.eqb_spec slot0 slot) as [-> | _]; [ | discriminate Hev ].
    destruct (lookup slot decls0) as [id0 | ] eqn:Hl0; [ | discriminate Hev ].
    destruct (lookup o decls0) as [id1 | ] eqn:Hl1; [ | discriminate Hev ].
    injection Hev as <-.
    pose proof Hrel0 as Hrel1. cbn [rel] in Hrel1.
    destruct Hrel1 as (sites & vs' & _ & -> & _).
    destruct Hd as ((Hacc & _) & Hslots).
    pose proof (rel_project_function_slot slot o decls0 vs0 vs' _ _ id0 id1
                  (Hslots _) Hl0 Hl1 Hacc Hrel0) as Hrel_x.
    assert (Hev' : eval_named statics' vars'
                     (Project_function_slot slot o (Var x0))
                   = Some (Value_closure_of_set o decls0 vs')).
    { cbn [eval_named eval_simple]. rewrite Hx0'.
      unfold project_function_slot. simpl.
      rewrite Pos.eqb_refl, Hl0, Hl1. reflexivity. }
    revert Ht.
    rewrite (rebuild_let_pending_eq en x _ slot o (Var x0) body _ Hep).
    intros ->. right. left. eexists. split.
    - rewrite step_let, Hev'. simpl kind_of_value. rewrite kind_eqb_refl.
      reflexivity.
    - split; [ exact Hflow' | ].
      eexists _, ce. simpl_state.
      split; [ exact Hbody_edges | split; [ reflexivity | ] ].
      split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
      + apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
        * intros _. eexists. split; [ reflexivity | exact Hrel_x ].
        * right. eexists. reflexivity.
      + split; [ exact Hheap | ].
        split; [ exact (conj Hexits (conj Hscr Hcf_body)) | ].
        assert (Hne' : Pos.eqb x0 x = false)
          by (apply Pos.eqb_neq; intros ->; exact (Hne eq_refl)).
        right. exists x0, slot, others, (Value_closure_of_set slot decls0 vs0),
          (Value_closure_of_set slot decls0 vs').
        split; [ reflexivity | ]. simpl. rewrite Hne'.
        split; [ exact Hx0 | split; [ exact Hx0' | split; [ exact Hrel0 | ] ] ].
        eauto. }
  destruct (named_plain d) eqn:Hplain.
  - revert Ht. rewrite (rebuild_let_eq en x _ d body Hep Hplain).
    destruct (is_dead_var sol x (kind_of_value v)) eqn:Hdead;
      [ congruence | ].
    destruct (is_var_used sol x (kind_of_value v)) eqn:Hused; intros ->.
    + right. left.
      destruct (named_rel en vars vars' statics statics' x d v Henv Hst Hvars
                  Hsflow Hfrag Hplain Hd Hev Hflows Hused)
        as (v' & Hev' & Hrel).
      eexists. split.
      * rewrite step_let, Hev', (rel_kind _ _ _ _ _ _ _ Hrel), kind_eqb_refl.
        reflexivity.
      * split; [ exact Hflow' | ].
        exists (bind_variables en [(x, kind_of_value v)]), ce. simpl_state.
        split; [ exact Hbody_edges | split; [ reflexivity | ] ].
        split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
        -- apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
           ++ intros _. exists v'. auto.
           ++ right. exists v'. reflexivity.
        -- split; [ exact Hheap | ].
           split; [ exact (conj Hexits (conj Hscr Hcf_body)) | ].
           left. reflexivity.
    + right. right. split; [ unfold measure; simpl; lia | ].
      split; [ exact Hflow' | ].
      exists (bind_variables en [(x, kind_of_value v)]), ce. simpl_state.
      split; [ exact Hbody_edges | split; [ reflexivity | ] ].
      split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
      * apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
        -- intros Hu. congruence.
        -- left. reflexivity.
      * split; [ exact Hheap | ].
        split; [ exact (conj Hexits (conj Hscr Hcf_body)) | ].
        left. reflexivity.
  - (* A closure of a set. *)
    destruct d as [ | | code vs | slot decls value_slots | | | ];
      try discriminate Hplain; simpl in Hfrag;
      [ unfold named_checked in Hfrag; destruct exact; discriminate Hfrag | ].
    simpl in Hev.
    destruct (lookup slot decls) as [id0 | ] eqn:Hid; [ | discriminate Hev ].
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    injection Hev as <-.
    set (bound := set_closure_nodes F (Node_variable x)).
    destruct (value_slots_rel en vars vars' bound value_slots vs Henv Hvs)
      as (vs' & Hvs' & Hempty & Hslots_rel).
    assert (Hev' : eval_named statics' vars'
                     (Closure_of_set slot decls
                        (rebuild_value_slots sol en bound value_slots))
                   = Some (Value_closure_of_set slot decls vs')).
    { simpl. rewrite Hid, Hvs'. reflexivity. }
    destruct (Hd id0 Hid) as (sites & Hsx & Hsites).
    assert (Hrel_x : rel (Value_closure_of_set slot decls vs) (Node_variable x)
                       (Value_closure_of_set slot decls vs')).
    { cbn [rel]. exists sites, vs'.
      assert (Hall : ∀ (f : function_slot) (id : code_id),
                lookup f decls = Some id ->
                may_come_from sol (sites f) (sites f)
                ∧ (∀ (g : function_slot) (id' : code_id),
                     lookup g decls = Some id' ->
                     g_constructor G (sites f) (Field_function_slot g)
                       (sites g))
                ∧ slots_related (slot_rel (sites f) vs') vs []).
      { intros f id Hf. destruct (Hsites f id Hf) as (Hdep & Hfs & Hws).
        split; [ exact (called_alloc is_local G sol cds Hc Hcds _ _
                          (Hdeps _ Hdep)) | ].
        split; [ exact Hfs | ].
        apply (Hslots_rel (sites f)); [ | exact Hws ].
        destruct (Hsites slot id0 Hid) as (_ & Hfs0 & _).
        pose proof (Hfs0 f id Hf) as Hcons. rewrite Hsx in Hcons.
        rewrite HG in Hcons. simpl in Hcons.
        unfold bound, set_closure_nodes. right.
        apply in_flat_map.
        exists (Node_variable x, Field_function_slot f, sites f).
        split; [ exact Hcons | ]. simpl. rewrite node_eqb_refl. left.
        reflexivity. }
      split; [ rewrite <- Hsx; exact (proj1 (Hall slot id0 Hid)) | ].
      split; [ reflexivity | split; [ | exact Hall ] ].
      intros He. apply Hempty. unfold named_checked in Hfrag.
      rewrite He in Hfrag. simpl in Hfrag.
      destruct value_slots; [ reflexivity | discriminate Hfrag ]. }
    pose proof (projections_live x slot decls vs id0 Hid Hflows
                  (other_slots slot decls) ce body
                  (other_slots_bound slot decls) Hbody_edges) as Hlive.
    revert Ht.
    rewrite (rebuild_let_set_eq en x _ slot decls value_slots body Hep).
    fold bound.
    match goal with
    | |- context [if ?b then Invalid else _] => destruct b eqn:Hdead
    end;
      [ rewrite Hnotdead, Hlive in Hdead; discriminate Hdead | ].
    match goal with
    | |- context [if ?b then Let _ _ _ _ else _] => destruct b eqn:Hused
    end;
      intros ->.
    + right. left. eexists. split.
      * rewrite step_let, Hev', kind_eqb_refl. reflexivity.
      * split; [ exact Hflow' | ].
        eexists _, ce. simpl_state.
        split; [ exact Hbody_edges | split; [ reflexivity | ] ].
        split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
        -- apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
           ++ intros _. eexists. split; [ reflexivity | exact Hrel_x ].
           ++ right. eexists. reflexivity.
        -- split; [ exact Hheap | ].
           split; [ exact (conj Hexits (conj Hscr Hcf_body)) | ].
           right. exists x, slot, (other_slots slot decls),
             (Value_closure_of_set slot decls vs),
             (Value_closure_of_set slot decls vs').
           split; [ reflexivity | ]. simpl. rewrite Pos.eqb_refl.
           split; [ reflexivity | split; [ reflexivity | ] ].
           split; [ exact Hrel_x | eauto ].
    + right. right. split; [ unfold measure; simpl; lia | ].
      split; [ exact Hflow' | ].
      eexists _, ce. simpl_state.
      split; [ exact Hbody_edges | split; [ reflexivity | ] ].
      split; [ | split; [ exact Hconts | split; [ exact Hst | ] ] ].
      * apply (env_ok_bind (env_kinds en) vars vars'); [ exact Henv | | ].
        -- intros Hu. apply orb_false_iff in Hused as [Hu' _]. congruence.
        -- left. reflexivity.
      * split; [ exact Hheap | ].
        split; [ exact (conj Hexits (conj Hscr Hcf_body)) | ].
        left. reflexivity.
Qed.

Lemma let_static_answered (bindings : list static_binding) (body : expr)
    (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (e' : expr) :
  related (State (Let_static bindings body) vars conts statics hp)
    (State e' vars' conts' statics' hp') ->
  answered related exits_related measure
    (State (Let_static bindings body) vars conts statics hp)
    (State e' vars' conts' statics' hp').
Proof.
  related_intro. destruct He as [Hbindings Hbody_edges].
  destruct Hok as (Hexits & Hscr & Hcf). simpl in Hexits, Hscr, Hcf.
  apply andb_true_iff in Hcf as [Hfrag Hcf_body].
  unfold answered. rewrite step_let_static in Hflow' |- *.
  destruct (eval_static_bindings vars bindings statics) as [statics2 | ]
    eqn:Hev; [ | exact I ].
  destruct (static_bindings_rel en vars vars' Henv bindings statics statics'
              statics2 Hfrag Hbindings Hst Hev)
    as (statics2' & Hev' & Hst2).
  rewrite rebuild_let_static_eq in Ht. subst e'.
  right. left. eexists. split.
  - rewrite step_let_static, Hev'. reflexivity.
  - split; [ exact Hflow' | ].
    exists (bind_variables en []), ce. simpl_state.
    split; [ exact Hbody_edges | split; [ reflexivity | ] ].
    split; [ exact Henv | ].
    split.
    + apply (conts_related_sub _ _ _ _ _ _ Hconts). intros k Hk. simpl.
      apply in_or_app. right. exact Hk.
    + split; [ exact Hst2 | split; [ exact Hheap | ] ].
      split; [ exact (conj Hexits (conj Hscr Hcf_body)) | left; reflexivity ].
Qed.

Lemma let_cont_non_recursive_answered (k : continuation)
    (h : continuation_handler) (body : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp hp' : heap)
    (e' : expr) :
  related (State (Let_cont_non_recursive k h body) vars conts statics hp)
    (State e' vars' conts' statics' hp') ->
  answered related exits_related measure
    (State (Let_cont_non_recursive k h body) vars conts statics hp)
    (State e' vars' conts' statics' hp').
Proof.
  related_intro. destruct He as [Hh_edges Hbody_edges].
  unfold answered. rewrite step_let_cont_non_recursive in Hflow' |- *.
  destruct Hok as (Hexits & Hscr & Hcf). simpl in Hexits, Hscr, Hcf.
  apply andb_true_iff in Hexits as [Hexits_h Hexits_body].
  destruct Hscr as [Hscr_h Hscr_body].
  apply andb_true_iff in Hcf as [Hcf_h Hcf_body].
  set (mask := handler_mask sol (handler_is_exn_handler h) h) in *.
  set (en' := bind_continuations en [(k, mask)]).
  rewrite rebuild_let_cont_non_recursive_eq in Ht. fold mask en' in Ht.
  destruct (among k (free_continuations_of_expr (rebuild en' body))) eqn:Hk;
    subst e'.
  - right. left. eexists. split; [ rewrite step_let_cont_non_recursive;
                                    reflexivity | ].
    split; [ exact Hflow' | ].
    exists en', ((k, handler_params h) :: ce). simpl_state.
    split; [ exact Hbody_edges | split; [ reflexivity | ] ].
    split; [ exact Henv | ].
    split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ].
    + intros j Hj. simpl in Hconts. simpl.
      destruct (Pos.eqb_spec j k) as [-> | Hne].
      * right. eexists _, _, _. split; [ reflexivity | ].
        split; [ reflexivity | ]. split; [ reflexivity | ].
        split; [ reflexivity | ].
        apply (Cont_rel_non_recursive _ _ _ _ _ _ ce).
        -- exact Henv.
        -- split; [ exact Hexits_h | split; assumption ].
        -- exact Hh_edges.
        -- apply (conts_related_sub _ _ _ _ _ _ Hconts). intros j' Hj'.
           apply in_or_app. left. exact Hj'.
      * apply Hconts. apply in_or_app. right.
        apply among_In. rewrite among_without. apply andb_true_intro.
        split; [ apply among_In; exact Hj | ].
        simpl. rewrite (proj2 (Pos.eqb_neq j k) Hne). reflexivity.
    + split; [ exact (conj Hexits_body (conj Hscr_body Hcf_body)) | ].
      left. reflexivity.
  - right. right. split; [ unfold measure; simpl; lia | ].
    split; [ exact Hflow' | ].
    exists en', ((k, handler_params h) :: ce). simpl_state.
    split; [ exact Hbody_edges | split; [ reflexivity | ] ].
    split; [ exact Henv | ].
    split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ].
    + apply conts_related_skip; [ exact Hconts | ].
      intros Hin. apply among_In in Hin. congruence.
    + split; [ exact (conj Hexits_body (conj Hscr_body Hcf_body)) | ].
      left. reflexivity.
Qed.

Lemma lookup_rebuilt_handlers (en : rebuild_env)
    (handlers : list (continuation * continuation_handler))
    (k : continuation) :
  lookup k (rebuilt_handlers en handlers)
  = match lookup k handlers with
    | Some h => Some (rebuild_h en false (handler_mask sol false h) h)
    | None => None
    end.
Proof.
  unfold rebuilt_handlers.
  exact (lookup_map_pairs
           (fun h => rebuild_h en false (handler_mask sol false h) h) k
           handlers).
Qed.

Lemma lookup_rec_env (en : rebuild_env)
    (handlers : list (continuation * continuation_handler))
    (k : continuation) :
  lookup k (env_conts (rec_env en handlers))
  = match lookup k handlers with
    | Some h => Some (handler_mask sol false h)
    | None => lookup k (env_conts en)
    end.
Proof.
  unfold rec_env, bind_continuations. cbn [env_conts].
  exact (lookup_map_pairs_app (fun h => handler_mask sol false h) k handlers
           (env_conts en)).
Qed.

Lemma lookup_rec_conts (ce : finite_map (list node))
    (handlers : list (continuation * continuation_handler))
    (k : continuation) :
  lookup k (rec_conts ce handlers)
  = match lookup k handlers with
    | Some h => Some (handler_params h)
    | None => lookup k ce
    end.
Proof.
  unfold rec_conts. exact (lookup_map_pairs_app handler_params k handlers ce).
Qed.

(* The continuations of a recursive group and of what is around it, in
   the two runs. *)
Lemma rec_conts_related (en : rebuild_env)
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env)
    (ce : finite_map (list node)) (ks : list continuation) :
  env_ok (env_kinds en) vars vars' ->
  (∀ (k' : continuation) (h' : continuation_handler),
     (k', h') ∈ handlers ->
     handler_ok h'
     ∧ handler_edges_ (rec_conts ce handlers) h'
     ∧ conts_related cont_rel
         (without (map fst handlers)
            (free_continuations_of_handler
               (rebuild_h (rec_env en handlers) false
                  (handler_mask sol false h') h')))
         conts conts' (env_conts en) ce) ->
  conts_related cont_rel (without (map fst handlers) ks) conts conts'
    (env_conts en) ce ->
  conts_related cont_rel ks
    (rec_closures handlers vars conts ++ conts)
    (rec_closures (rebuilt_handlers (rec_env en handlers) handlers) vars'
       conts' ++ conts')
    (env_conts (rec_env en handlers)) (rec_conts ce handlers).
Proof.
  intros Henv Hall Hks j Hj.
  rewrite lookup_rec_closures, lookup_rec_closures, lookup_rebuilt_handlers,
    lookup_rec_env, lookup_rec_conts.
  destruct (lookup j handlers) as [hj | ] eqn:Hlj.
  - right. eexists _, _, _. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    exact (Cont_rel_recursive j hj en vars vars' conts conts' handlers ce Henv
             (lookup_in j hj handlers Hlj) Hall).
  - apply Hks. apply among_In. rewrite among_without.
    replace (among j (map fst handlers)) with false
      by (symmetry; exact (lookup_none_among j handlers Hlj)).
    simpl. rewrite andb_true_r. apply among_In. exact Hj.
Qed.

Lemma let_cont_recursive_answered
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (e' : expr) :
  related (State (Let_cont_recursive handlers body) vars conts statics hp)
    (State e' vars' conts' statics' hp') ->
  answered related exits_related measure
    (State (Let_cont_recursive handlers body) vars conts statics hp)
    (State e' vars' conts' statics' hp').
Proof.
  related_intro. destruct He as [Hhandlers Hbody_edges].
  unfold answered. rewrite step_let_cont_recursive in Hflow' |- *.
  destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
  destruct Hok as (Hexits & Hscr & Hcf). simpl in Hexits, Hscr, Hcf.
  apply andb_true_iff in Hexits as [Hexits_h Hexits_body].
  destruct Hscr as [Hscr_h Hscr_body].
  apply andb_true_iff in Hcf as [Hcf_h Hcf_body].
  rewrite forallb_forall in Hexits_h, Hcf_h.
  rewrite rebuild_let_cont_recursive_eq in Ht. subst e'.
  set (en' := rec_env en handlers) in *.
  right. left. eexists. split.
  - rewrite step_let_cont_recursive.
    assert (Hbinds' : handlers_bind_statics (rebuilt_handlers en' handlers)
                      = false).
    { apply not_true_iff_false. intros Hb. unfold handlers_bind_statics in Hb.
      apply existsb_exists in Hb as ([k h'] & Hin & Hb).
      unfold rebuilt_handlers in Hin.
      apply in_map_iff in Hin as ([k' h] & [= <- <-] & Hin).
      pose proof (proj2 binds_statics_rebuild h (Hcf_h (k', h) Hin) en' false
                    (handler_mask sol false h) Hb) as Hb'.
      unfold handlers_bind_statics in Hbinds.
      rewrite (proj2 (existsb_exists _ _) (ex_intro _ (k', h) (conj Hin Hb')))
        in Hbinds.
      discriminate Hbinds. }
    rewrite Hbinds'. reflexivity.
  - split; [ exact Hflow' | ].
    exists en', (rec_conts ce handlers). simpl_state.
    split; [ exact Hbody_edges | split; [ reflexivity | ] ].
    split; [ exact Henv | ].
    split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ].
    + simpl in Hconts. rewrite map_fst_rebuilt_handlers in Hconts.
      apply rec_conts_related; [ exact Henv | | ].
      * intros k' h' Hin. split; [ | split ].
        -- split; [ exact (Hexits_h (k', h') Hin) | ].
           split; [ exact (proj1 (every_in _ _ _) Hscr_h (k', h') Hin) | ].
           exact (Hcf_h (k', h') Hin).
        -- exact (proj1 (every_in _ _ _) Hhandlers (k', h') Hin).
        -- apply (conts_related_sub _ _ _ _ _ _ Hconts). intros j Hj.
           apply among_In. apply among_In in Hj.
           rewrite among_without in Hj |- *.
           apply andb_true_iff in Hj as [Hj Hnot].
           rewrite Hnot, andb_true_r. apply among_In. apply in_or_app. left.
           apply in_flat_map.
           exists (k', rebuild_h en' false (handler_mask sol false h') h').
           split; [ | apply among_In; exact Hj ].
           unfold rebuilt_handlers. apply in_map_iff. exists (k', h').
           auto.
      * apply (conts_related_sub _ _ _ _ _ _ Hconts). intros j Hj.
        apply among_In. apply among_In in Hj.
        rewrite among_without in Hj |- *.
        apply andb_true_iff in Hj as [Hj Hnot].
        rewrite Hnot, andb_true_r. apply among_In. apply in_or_app. right.
        apply among_In. exact Hj.
    + split; [ exact (conj Hexits_body (conj Hscr_body Hcf_body)) | ].
      left. reflexivity.
Qed.

(* A continuation of the unit has one parameter, at a node with any
   usage. *)
Lemma unit_conts_lookup (k : continuation) :
  unit_cont k ->
  ∃ p : node, lookup k (unit_conts u) = Some [p] ∧ p ∈ unit_cont_nodes u k.
Proof.
  intros Hk. unfold unit_cont_nodes.
  destruct (lookup k (unit_conts u)) as [ps | ] eqn:Hl.
  - unfold unit_conts in Hl. simpl in Hl.
    destruct (Pos.eqb k (unit_exn u)).
    + injection Hl as <-. exists (Node_made 2).
      split; [ reflexivity | left; reflexivity ].
    + destruct (Pos.eqb k (unit_return u)); [ | discriminate Hl ].
      injection Hl as <-. exists (Node_made 1).
      split; [ reflexivity | left; reflexivity ].
  - exfalso. unfold unit_conts in Hl. simpl in Hl.
    destruct Hk as [-> | ->]; rewrite Pos.eqb_refl in Hl;
      [ destruct (Pos.eqb (unit_return u) (unit_exn u)) | ]; discriminate Hl.
Qed.

Lemma apply_cont_answered (ac : apply_cont) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp hp' : heap)
    (e' : expr) :
  related (State (Apply_cont ac) vars conts statics hp)
    (State e' vars' conts' statics' hp') ->
  answered related exits_related measure
    (State (Apply_cont ac) vars conts statics hp)
    (State e' vars' conts' statics' hp').
Proof.
  related_intro. pose proof He as Hedges. destruct He as [Haliases Hnoted].
  unfold answered. rewrite step_apply_cont in Hflow' |- *.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs; [ | exact I ].
  change (rebuild en (Apply_cont ac)) with (rebuild_apply_cont sol en ac)
    in Ht.
  unfold rebuild_apply_cont in Ht.
  rewrite (rewrite_apply_cont_some en ac
             (args_have_sources vars _ vs Hvars Hnoted Hvs)) in Ht.
  subst e'.
  destruct (rewrite_simples_eval en vars vars' Henv (ac_args ac) vs Hnoted Hvs)
    as (vs' & Hvs' & Hkinds & Hrels).
  assert (Hk : ac_continuation ac
               ∈ free_continuations_of_expr
                   (Apply_cont (rebuilt_apply_cont en ac)))
    by (left; reflexivity).
  rewrite step_apply_cont. cbn [ac_args ac_continuation rebuilt_apply_cont].
  unfold apply_continuation, resolve in Hflow' |- *.
  destruct (Hconts _ Hk)
    as [(Hnone & Hnone' & Hmnone & Hunit_k & Hce)
       | (mask & c & c' & Hmask & Hc_ & Hc'_ & Hce & Hcrel)].
  - (* An exit: one value, at a node with any usage. *)
    rewrite Hmnone, Hvs', Hnone, Hnone'. cbn [apply_cont_value].
    rewrite Hheap. eexists. split; [ reflexivity | ].
    destruct Hok as [Hone _]. cbn [exits_one_value] in Hone.
    unfold apply_cont_exits_one_value in Hone.
    replace (among (ac_continuation ac) [unit_return u; unit_exn u])
      with true in Hone
      by (symmetry; apply among_In;
          destruct Hunit_k as [-> | ->]; [ left | right; left ]; reflexivity).
    apply Nat.eqb_eq in Hone.
    destruct (unit_conts_lookup _ Hunit_k) as (p & Hp & Hpin).
    rewrite <- Hce in Hp.
    destruct (ac_args ac) as [ | a [ | a2 args]] eqn:Hargs;
      try discriminate Hone.
    simpl in Hvs. destruct (eval_simple vars a) as [w | ] eqn:Ha;
      [ | discriminate Hvs ].
    injection Hvs as <-.
    simpl in Hvs'.
    destruct (eval_simple vars' (rewrite_simple sol en a)) as [w' | ] eqn:Ha';
      [ | discriminate Hvs' ].
    injection Hvs' as <-.
    pose proof (alias_input _ _ _ Hc _ _
                  (Haliases [p] Hp p a (or_introl eq_refl))) as Hal.
    pose proof (any_usage_alias _ _ _ Hc _ _
                  (any_usage_input _ _ _ Hc _ (Hunit _ p Hpin)) Hal) as Hau.
    pose proof (Hrels 0 a w w' eq_refl eq_refl eq_refl
                  (kept_of_hu en a (au_hu _ _ _ Hc _ Hau))) as Hrel.
    destruct (sim_observed sol (exact = true) w w'
                (rel_au_sim G sol keep (exact = true) is_local Hc Hlocal _ _ _
                   Hau Hrel))
      as [Ho He].
    split; [ split; [ exact Ho | exact He ] | exact I ].
  - (* A handler. *)
    rewrite Hmask, (eval_simples_filter_kept vars' mask _ _ Hvs'), Hc_, Hc'_.
    rewrite Hc_ in Hflow'.
    revert Hmask Hc_ Hc'_ Hce.
    destruct Hcrel as [h en0 vars1 vars1' conts1 conts1' ce0 Henv0 Hok0
                         Hedges0 Hconts0
                      | j h en0 vars1 vars1' conts1 conts1' handlers ce0 Henv0
                          Hin0 Hall0];
      intros Hmask Hc_ Hc'_ Hce;
      destruct h as [params metadata handler];
      cbn [apply_cont_value] in Hflow' |- *;
      destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
        eqn:Hkb; try exact I;
      apply kinds_eqb_eq in Hkb;
      destruct (bind_params params vs vars1) as [hv | ] eqn:Hbind;
        try exact I;
      right; left;
      destruct (flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ Hflow')
        as [Hhv _];
      simpl_state;
      pose proof (params_not_dead params (hm_params_kinds metadata) vs vars1 hv
                    Hbind Hhv) as Hnodead;
      pose proof (apply_cont_args_rel en vars vars' ce ac params vs vs' Henv
                    Hedges Hce Hvs Hrels) as Hargs;
      rewrite rebuild_handler_eq.
    + rewrite Hnodead. cbn [andb].
      destruct (enter_handler
                  (handler_is_exn_handler (Handler params metadata handler))
                  params metadata handler
                  (rebuild (bind_variables en0
                              (with_kinds params (hm_params_kinds metadata)))
                     handler)
                  en0 vars1 vars1' hv conts1' [] vs vs' statics' hp' Henv0 Hkb
                  Hkinds Hbind Hargs)
        as (hv' & Hstep & Hok').
      eexists. split; [ exact Hstep | ].
      split; [ exact Hflow' | ].
      eexists _, ce0. simpl_state.
      split; [ exact Hedges0 | split; [ reflexivity | ] ].
      split; [ exact Hok' | ].
      split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ];
        [ | split; [ exact Hok0 | left; reflexivity ] ].
      rewrite rebuild_handler_eq, Hnodead in Hconts0. exact Hconts0.
    + cbn [andb].
      destruct (enter_handler false params metadata handler
                  (rebuild (bind_variables (rec_env en0 handlers)
                              (with_kinds params (hm_params_kinds metadata)))
                     handler)
                  (rec_env en0 handlers) vars1 vars1' hv conts1'
                  (rebuilt_handlers (rec_env en0 handlers) handlers) vs vs'
                  statics' hp' Henv0 Hkb Hkinds Hbind Hargs)
        as (hv' & Hstep & Hok').
      eexists. split; [ exact Hstep | ].
      split; [ exact Hflow' | ].
      destruct (Hall0 j _ Hin0) as (Hok_h & Hedges_h & Hconts_h).
      eexists _, (rec_conts ce0 handlers). simpl_state.
      split; [ exact Hedges_h | split; [ reflexivity | ] ].
      split; [ exact Hok' | ].
      split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ];
        [ | split; [ exact Hok_h | left; reflexivity ] ].
      rewrite rebuild_handler_eq in Hconts_h. cbn [andb] in Hconts_h.
      exact (rec_conts_related en0 handlers vars1 vars1' conts1 conts1' ce0 _
               Henv0 Hall0 Hconts_h).
Qed.

Lemma find_arm_in (i : Z) (arms : list (Z * apply_cont)) (ac : apply_cont) :
  find_arm i arms = Some ac -> (i, ac) ∈ arms.
Proof.
  induction arms as [ | [d ac'] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb_spec i d) as [<- | _].
  - intros [= <-]. left. reflexivity.
  - intros H. right. exact (IH H).
Qed.

Lemma find_arm_lookup (i : Z) (arms : list (Z * apply_cont)) :
  find_arm i arms = Keyed.lookup Z.eqb i arms.
Proof.
  induction arms as [ | [d ac] arms IH]; simpl; [ reflexivity | ].
  destruct (Z.eqb i d); [ reflexivity | exact IH ].
Qed.

Lemma find_arm_compact_eq (i : Z) (arms : list (Z * apply_cont)) :
  find_arm i (Keyed.compact Z.eqb arms) = find_arm i arms.
Proof.
  rewrite !find_arm_lookup.
  exact (keyed_lookup_compact Z Z.eqb z_eqb_decides_equality _ i arms).
Qed.

(* The arm found among the arms that a map keeps or drops, keeping their
   discriminants. *)
Lemma find_arm_flat_map (f : Z * apply_cont -> list (Z * apply_cont))
    (i : Z) (arms : list (Z * apply_cont)) (ac ac' : apply_cont) :
  (∀ (d : Z) (a : apply_cont),
     f (d, a) = [] ∨ ∃ a' : apply_cont, f (d, a) = [(d, a')]) ->
  find_arm i arms = Some ac -> f (i, ac) = [(i, ac')] ->
  find_arm i (flat_map f arms) = Some ac'.
Proof.
  intros Hf. induction arms as [ | [d a] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb_spec i d) as [<- | Hne].
  - intros [= <-] Hfi. rewrite Hfi. simpl. rewrite Z.eqb_refl. reflexivity.
  - intros Harm Hfi.
    destruct (Hf d a) as [-> | (a' & ->)]; simpl; [ exact (IH Harm Hfi) | ].
    destruct (Z.eqb_spec i d); [ contradiction | exact (IH Harm Hfi) ].
Qed.

Definition rebuilt_arms (en : rebuild_env) (arms : list (Z * apply_cont))
    : list (Z * apply_cont) :=
  flat_map
    (fun '(d, ac) =>
       match rewrite_apply_cont sol en ac with
       | Some ac => [(d, ac)]
       | None => []
       end)
    (Keyed.compact Z.eqb arms).

Lemma rebuild_switch_eq (en : rebuild_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  rebuild en (Switch scrutinee arms)
  = if Keyed.is_empty (rebuilt_arms en arms) then Invalid
    else Switch (rewrite_simple sol en scrutinee) (rebuilt_arms en arms).
Proof. reflexivity. Qed.

Lemma switch_answered (scrutinee : simple) (arms : list (Z * apply_cont))
    (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (e' : expr) :
  related (State (Switch scrutinee arms) vars conts statics hp)
    (State e' vars' conts' statics' hp') ->
  answered related exits_related measure
    (State (Switch scrutinee arms) vars conts statics hp)
    (State e' vars' conts' statics' hp').
Proof.
  related_intro.
  unfold answered. rewrite step_switch in Hflow' |- *.
  destruct (eval_simple vars scrutinee) as [w | ] eqn:Hw; [ | exact I ].
  destruct (naked_immediate_of_value w) as [i | ] eqn:Hi; [ | exact I ].
  destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
  pose proof (proj1 (every_in _ _ _) He (i, ac) (find_arm_compact _ _ _ Harm))
    as Hac.
  destruct Hok as (Hexits & Hscr & _). cbn [exits_one_value scrutinees_used]
    in Hexits, Hscr.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs.
  2: { left. rewrite step_apply_cont, Hvs. reflexivity. }
  right. left.
  pose proof (rewrite_apply_cont_some en ac
                (args_have_sources vars _ vs Hvars (proj2 Hac) Hvs)) as Hrw.
  assert (Harm' : find_arm i (rebuilt_arms en arms)
                  = Some (rebuilt_apply_cont en ac)).
  { apply (find_arm_flat_map _ i _ ac).
    - intros d a. destruct (rewrite_apply_cont sol en a);
        [ right; eexists; reflexivity | left; reflexivity ].
    - rewrite find_arm_compact_eq. exact Harm.
    - cbv beta iota. rewrite Hrw. reflexivity. }
  rewrite rebuild_switch_eq in Ht.
  destruct (rebuilt_arms en arms) as [ | arm arms'] eqn:Harms;
    [ discriminate Harm' | ].
  rewrite <- Harms in Ht, Harm'. cbn [Keyed.is_empty] in Ht.
  rewrite Harms in Ht. cbn [Keyed.is_empty] in Ht. rewrite <- Harms in Ht.
  subst e'.
  destruct (rewrite_simple_au en vars vars' scrutinee w Henv
              (or_intror (any_usage_input _ _ _ Hc _ Hscr)) Hw)
    as (w' & Hw' & Hel).
  assert (Hww : w' = w).
  { apply element_rel_sim in Hel.
    destruct w; simpl in Hi; try discriminate Hi; exact Hel. }
  subst w'.
  eexists. split.
  - rewrite step_switch, Hw', Hi, Harm'. reflexivity.
  - split; [ exact Hflow' | ].
    exists en, ce. simpl_state. split; [ exact Hac | ]. split.
    + change (rebuild en (Apply_cont ac)) with (rebuild_apply_cont sol en ac).
      unfold rebuild_apply_cont. rewrite Hrw. reflexivity.
    + split; [ exact Henv | ].
      split; [ | split; [ exact Hst | split; [ exact Hheap | ] ] ].
      * apply (conts_related_sub _ _ _ _ _ _ Hconts). intros j Hj. simpl.
        apply in_flat_map. exists (i, rebuilt_apply_cont en ac).
        split; [ exact (find_arm_in _ _ _ Harm') | exact Hj ].
      * split; [ | left; apply (pending_not_let en vars vars' _ Hpend);
                   intros ? ? [ | ? ? ]; reflexivity ].
        split; [ | split; [ exact I | reflexivity ] ]. cbn [exits_one_value].
        rewrite forallb_forall in Hexits.
        exact (Hexits (i, ac) (find_arm_in _ _ _ Harm)).
Qed.

(* A step of the first run is answered by the second. *)
Lemma related_answered (s t : state) :
  related s t -> answered related exits_related measure s t.
Proof.
  rewrite (state_eta s), (state_eta t).
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros Hrel.
  - exact (let_answered _ _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (let_static_answered _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (let_cont_non_recursive_answered _ _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (let_cont_recursive_answered _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - destruct Hrel as [_ (en & ce & _ & _ & _ & _ & _ & _ & (_ & _ & Hcf) & _)].
    simpl_state. discriminate Hcf.
  - exact (apply_cont_answered _ _ _ _ _ _ _ _ _ _ Hrel).
  - exact (switch_answered _ _ _ _ _ _ _ _ _ _ _ Hrel).
  - unfold answered. rewrite step_invalid. exact I.
Qed.

(* The unit refines its rebuild, given what the traversal makes of it and
   what the unit asks of it. *)
Lemma unit_refines (e : expr) :
  edges (unit_conts u) e ->
  (∀ k : continuation, k ∈ free_continuations_of_expr e -> unit_cont k) ->
  expr_ok e ->
  refines_observably (State e [] [] empty_static_env empty_heap)
    (State (rebuild FlambdaReaperRebuild.empty_env e) [] [] empty_static_env
       empty_heap)
  ∧ (exact = true ->
     refines (State e [] [] empty_static_env empty_heap)
       (State (rebuild FlambdaReaperRebuild.empty_env e) [] []
          empty_static_env empty_heap)).
Proof.
  intros He Hfree Hok.
  enough (Hrel : related (State e [] [] empty_static_env empty_heap)
                   (State (rebuild FlambdaReaperRebuild.empty_env e) [] []
                      empty_static_env empty_heap)).
  { split.
    - exact (shrinking_refines_observably related exits_related measure
               related_answered _ _ exits_related_observed Hrel).
    - intros Hx.
      exact (shrinking_refines related exits_related measure related_answered
               _ _ (fun vs vs' => exits_related_eq vs vs' Hx) Hrel). }
  split; [ exact (initial_flow_invariant is_local G sol noted deps cds keep
                    codes symbols u e He) | ].
  exists FlambdaReaperRebuild.empty_env, (unit_conts u). simpl_state.
  split; [ exact He | split; [ reflexivity | ] ].
  split; [ intros x v Hx; discriminate Hx | ].
  split.
  - intros k Hk. left.
    split; [ reflexivity | split; [ reflexivity | split; [ reflexivity | ] ] ].
    split; [ | reflexivity ].
    exact (Hfree k (proj1 free_rebuild e (proj2 (proj2 Hok)) _ k Hk)).
  - split.
    + split; [ intros s _; reflexivity | ].
      split; [ intros s tag fields Hl; discriminate Hl | ].
      split; [ intros s allocated Hl; discriminate Hl | ].
      intros id. left. reflexivity.
    + split; [ reflexivity | split; [ exact Hok | left; reflexivity ] ].
Qed.

End Run.

(* ======================================================================== *)
(* The facts that the traversal makes of the [Switch]es                     *)
(* ======================================================================== *)

Lemma flat_map_nil {A B : Type} (l : list A) :
  flat_map (fun _ : A => @nil B) l = [].
Proof. induction l as [ | a l IH]; [ reflexivity | exact IH ]. Qed.

(* A step of the traversal only adds to what it is given, and makes [P]
   of what follows. *)
Definition grows (step : acc -> acc) (P : acc -> Prop) : Prop :=
  (∀ a : acc, acc_le a (step a)) ∧ ∀ a a' : acc, acc_le (step a) a' -> P a'.

Lemma makes_grows (ids : list code_id) (step : acc -> acc) (P : acc -> Prop) :
  makes ids step P -> grows step P.
Proof.
  intros H. split; [ intros a; exact (proj1 (proj1 (H a))) | ].
  intros a a' Hle. exact (proj2 (H a) a' Hle).
Qed.

Lemma grows_compose (step step' : acc -> acc) (P P' : acc -> Prop) :
  grows step P -> grows step' P' ->
  grows (fun a => step' (step a)) (fun a' => P a' ∧ P' a').
Proof.
  intros [H1 H2] [H1' H2']. split.
  - intros a. exact (acc_le_trans _ _ _ (H1 a) (H1' (step a))).
  - intros a a' Hle. split; [ | exact (H2' (step a) a' Hle) ].
    apply (H2 a). exact (acc_le_trans _ _ _ (H1' (step a)) Hle).
Qed.

Lemma grows_weaken (step : acc -> acc) (P P' : acc -> Prop) :
  (∀ a : acc, P a -> P' a) -> grows step P -> grows step P'.
Proof.
  intros H [H1 H2]. split; [ exact H1 | ].
  intros a a' Hle. exact (H _ (H2 a a' Hle)).
Qed.

Lemma grows_ext (step step' : acc -> acc) (P : acc -> Prop) :
  (∀ a : acc, step' a = step a) -> grows step P -> grows step' P.
Proof.
  intros He [H1 H2]. split.
  - intros a. rewrite He. exact (H1 a).
  - intros a a' Hle. rewrite He in Hle. exact (H2 a a' Hle).
Qed.

Lemma grows_fold {A : Type} (f : acc -> A -> acc) (P : A -> acc -> Prop)
    (l : list A) :
  (∀ x : A, x ∈ l -> grows (fun a => f a x) (P x)) ->
  grows (fun a => List.fold_left f l a) (fun a' => ∀ x : A, x ∈ l -> P x a').
Proof.
  induction l as [ | x l IH]; intros H.
  - split; [ intros a; apply acc_le_refl | intros a a' _ x [] ].
  - pose proof (grows_compose _ _ _ _ (H x (or_introl eq_refl))
                  (IH (fun y Hy => H y (or_intror Hy)))) as Hc.
    simpl. eapply grows_weaken; [ | exact Hc ].
    intros a [Hx Hl] y [<- | Hy]; [ exact Hx | exact (Hl y Hy) ].
Qed.

(* At toplevel, the scrutinee of every [Switch] of a term without calls
   has any usage ([cond_any_usage]). *)
Lemma traverse_scrutinees (ex : bool) :
  (∀ e : expr,
     checked ex e = true ->
     ∀ env : traverse_env,
       (∀ w : value_slot,
          w ∈ projected_value_slots e -> should_keep_value_slot env w = true) ->
       grows (traverse env e)
         (fun a' =>
            env_current_code_id env = None -> scrutinees_used (acc_graph a') e))
  ∧ (∀ h : continuation_handler,
       handler_checked ex h = true ->
       ∀ env : traverse_env,
         (∀ w : value_slot,
            w ∈ handler_projected_value_slots h ->
            should_keep_value_slot env w = true) ->
         grows (traverse_handler env h)
           (fun a' =>
              env_current_code_id env = None ->
              handler_scrutinees_used (acc_graph a') h)).
Proof.
  apply expr_handler_ind.
  - (* [Let] *)
    intros x kind d body IH Hcf env Hkeep. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hd Hbody].
    apply named_checked_no_code in Hd.
    assert (Hd' : named_in_flow_fragment d = true)
      by (destruct d; try discriminate Hd; reflexivity).
    pose proof (grows_compose _ _ _ _
                  (makes_grows _ _ _
                     (traverse_named_makes env True x d Hd' (fun _ => I)
                        (fun w Hw =>
                           Hkeep w (in_or_app _ _ _ (or_introl Hw)))))
                  (IH Hbody env
                     (fun w Hw => Hkeep w (in_or_app _ _ _ (or_intror Hw)))))
      as Hboth.
    eapply grows_weaken; [ | exact Hboth ]. intros a' [_ H]. exact H.
  - (* [Let_static] *)
    intros bindings body Hcodes IH Hcf env Hkeep. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hbindings Hbody].
    rewrite forallb_forall in Hbindings.
    assert (Hprep :
      grows (fun a => List.fold_left (fun a b => prepare_binding b a) bindings
                        a)
        (fun _ => True)).
    { eapply grows_weaken;
        [ | apply (grows_fold _ (fun _ _ => True)) ]; [ intros; exact I | ].
      intros b _. split; [ | intros; exact I ].
      intros a. destruct b as [id c | id | s const]; simpl;
        [ | apply acc_le_refl | apply acc_le_refl ].
      destruct (prepare_code_spec id c a) as (cd & _ & Hle & _). exact Hle. }
    assert (Hstat :
      grows (fun a =>
               List.fold_left (fun a b => traverse_static_binding env b a)
                 bindings a)
        (fun _ => True)).
    { eapply grows_weaken;
        [ | apply (grows_fold _ (fun _ _ => True)) ]; [ intros; exact I | ].
      intros b Hin. specialize (Hbindings b Hin).
      destruct (binding_checked_cases _ _ Hbindings)
        as [Hnc | (id & c & -> & _ & _)];
        [ | split; [ intros a; apply acc_le_refl | intros; exact I ] ].
      apply (makes_grows []).
      destruct b as [ | id | s [ | slot syms decls vs | t fields | | | ]];
        simpl in Hnc; try discriminate Hnc.
      + apply makes_steady; [ intros a; apply steady_refl | intros; exact I ].
      + eapply makes_weaken;
          [ | apply (static_set_makes env (map snd syms)) ];
          [ intros; exact I | ].
        apply andb_true_iff in Hnc as [Hbound _].
        rewrite Hbound. simpl. apply forallb_forall. intros [g sym] Hgs.
        simpl. apply among_In. apply in_map_iff. exists (g, sym). auto.
      + eapply makes_weaken;
          [ | exact (block_deps_makes (Node_symbol s) fields) ].
        intros; exact I. }
    assert (Hbind :
      grows (fun a => List.fold_left (fun a b => traverse_binding env b a)
                        bindings a)
        (fun _ => True)).
    { eapply grows_weaken;
        [ | apply (grows_fold _ (fun _ _ => True)) ]; [ intros; exact I | ].
      intros b Hin. pose proof (Hbindings b Hin) as Hb.
      split; [ | intros; exact I ].
      destruct b as [id c | id | s const]; simpl;
        [ | intros a; apply acc_le_refl | intros a; apply acc_le_refl ].
      destruct (binding_checked_cases _ _ Hb)
        as [Hnc | (id' & c' & Heq & He & Hnc)]; [ discriminate Hnc | ].
      injection Heq as <- <-.
      assert (Hcb : checked ex (code_body c) = true)
        by (rewrite He; exact (no_code_checked _ Hnc)).
      intros a. pose proof (Hcodes id c Hin Hcb) as IHc.
      destruct c as [params mc ret exn md cbody]. simpl.
      destruct (lookup id (acc_code_deps a)) as [cd | ];
        [ | apply acc_le_refl ].
      match goal with
      | |- acc_le a (traverse _ _ (add_facts ?l a)) =>
        apply (acc_le_trans _ (add_facts l a));
          [ exact (proj1 (add_facts_steady l a)) | ]
      end.
      assert (Hk : ∀ w : value_slot,
                w ∈ projected_value_slots cbody ->
                should_keep_value_slot (code_env env id ret exn cd) w = true).
      { intros w Hw. apply Hkeep. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id (Code params mc ret exn md cbody)).
        split; [ exact Hin | exact Hw ]. }
      exact (proj1 (IHc _ Hk) _). }
    pose proof (grows_compose _ _ _ _
                  (grows_compose _ _ _ _ (grows_compose _ _ _ _ Hprep Hstat)
                     Hbind)
                  (IH Hbody env
                     (fun w Hw => Hkeep w (in_or_app _ _ _ (or_intror Hw)))))
      as Hall.
    eapply grows_ext; [ | eapply grows_weaken; [ | exact Hall ] ];
      [ intros a; reflexivity | ].
    intros a' [_ H]. exact H.
  - (* [Let_cont_non_recursive] *)
    intros k h body IHh IH Hcf env Hkeep. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hh Hbody].
    pose proof (grows_compose _ _ _ _
                  (grows_compose _ _ _ _
                     (IHh Hh env
                        (fun w Hw => Hkeep w (in_or_app _ _ _ (or_introl Hw))))
                     (makes_grows _ _ _
                        (makes_steady _ (fun _ => True)
                           (exn_handler_deps_steady env h)
                           (fun _ _ _ => I))))
                  (IH Hbody (add_cont env k (handler_params h))
                     (fun w Hw => Hkeep w (in_or_app _ _ _ (or_intror Hw)))))
      as Hboth.
    eapply grows_weaken; [ | exact Hboth ].
    intros a' [[Hh' _] Hb] Henv. split; [ exact (Hh' Henv) | exact (Hb Henv) ].
  - (* [Let_cont_recursive] *)
    intros handlers body IHh IH Hcf env Hkeep. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    set (env' := add_rec_conts env handlers).
    assert (Hcode' : env_current_code_id env' = env_current_code_id env)
      by exact (proj1 (add_rec_conts_fields env handlers)).
    assert (Hkeep' : ∀ w : value_slot,
              w ∈ projected_value_slots (Let_cont_recursive handlers body) ->
              should_keep_value_slot env' w = true).
    { intros w Hw. unfold should_keep_value_slot, env'.
      rewrite (proj2 (add_rec_conts_fields env handlers)). exact (Hkeep w Hw). }
    assert (Hfold :
      grows
        (fun a =>
           List.fold_left (fun a '(_, h) => traverse_handler env' h a)
             handlers a)
        (fun a' =>
           ∀ kh, kh ∈ handlers ->
             let '(_, h) := kh in
             env_current_code_id env' = None ->
             handler_scrutinees_used (acc_graph a') h)).
    { apply (grows_fold _
               (fun kh a' =>
                  let '(_, h) := kh in
                  env_current_code_id env' = None ->
                  handler_scrutinees_used (acc_graph a') h)).
      intros [k h] Hin.
      apply (IHh k h Hin (Hhandlers _ Hin) env').
      intros w Hw. apply Hkeep'. apply in_or_app. left.
      apply in_flat_map. exists (k, h). auto. }
    pose proof (grows_compose _ _ _ _ Hfold
                  (IH Hbody env'
                     (fun w Hw => Hkeep' w (in_or_app _ _ _ (or_intror Hw)))))
      as Hboth.
    eapply grows_ext; [ | eapply grows_weaken; [ | exact Hboth ] ];
      [ intros a; reflexivity | ].
    intros a' [Hall Hb] Henv. rewrite <- Hcode' in Henv.
    split; [ | exact (Hb Henv) ].
    apply (every_in _ _ _). intros [k h] Hin. exact (Hall (k, h) Hin Henv).
  - (* [Apply] *)
    intros ap Hcf. discriminate Hcf.
  - (* [Apply_cont] *)
    intros ac _ env _.
    eapply grows_weaken;
      [ | apply (makes_grows []); apply apply_cont_deps_makes ].
    intros; exact I.
  - (* [Switch] *)
    intros scrutinee arms _ env _.
    assert (Hcond :
      grows (cond_any_usage env scrutinee)
        (fun a' => env_current_code_id env = None ->
           g_any_usage (acc_graph a') (node_of_simple scrutinee))).
    { apply (makes_grows []). apply makes_steady;
        [ apply cond_any_usage_steady | ].
      intros a a' Hle Henv. apply facts_any_usage.
      unfold cond_any_usage in Hle. rewrite Henv in Hle.
      exact (add_facts_later _ _ _ _ Hle (or_introl eq_refl)). }
    assert (Hfold :
      grows
        (fun a =>
           List.fold_left (fun a '(_, ac) => apply_cont_deps env ac a)
             (Keyed.compact Z.eqb arms) a)
        (fun _ => True)).
    { eapply grows_weaken;
        [ | apply (grows_fold _ (fun _ _ => True)) ]; [ intros; exact I | ].
      intros [d ac] _.
      eapply grows_weaken;
        [ | apply (makes_grows []); apply apply_cont_deps_makes ].
      intros; exact I. }
    pose proof (grows_compose _ _ _ _ Hcond Hfold) as Hboth.
    eapply grows_weaken; [ | exact Hboth ]. intros a' [H _]. exact H.
  - (* [Invalid] *)
    intros _ env _.
    split; [ intros a; apply acc_le_refl | intros; exact I ].
  - (* A handler *)
    intros params metadata handler IH Hcf env Hkeep.
    exact (IH Hcf env Hkeep).
Qed.

Lemma traverse_unit_scrutinees (ex : bool)
    (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr) :
  checked ex e = true ->
  scrutinees_used (graph_of_facts (traverse_unit flag u e)) e.
Proof.
  intros Hcf. rewrite traverse_unit_eq.
  destruct (proj1 (traverse_scrutinees ex) e Hcf (initial_env flag u e)
              (fun w Hw => proj2 (among_In _ _) Hw))
    as [_ Hscr].
  set (a4 := traverse (initial_env flag u e) e initial_acc) in *.
  set (a5 := add_fact (Fact_any_usage (Node_made 1)) a4).
  destruct (resolve_delayed_deps_spec (static_names symbols_of_binding e) a5)
    as (S6 & _).
  exact (Hscr initial_acc _
           (proj1 (steady_trans _ _ _ (add_facts_steady _ a4) S6)) eq_refl).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit without closures meets [no_code_no_slots]. *)
Theorem closure_free_no_code_no_slots :
  closure_free_no_code_no_slots_statement.
Proof.
  enough (H : (∀ e : expr, closure_free e = true -> no_code_no_slots e = true)
              ∧ (∀ h : continuation_handler,
                   handler_closure_free h = true ->
                   handler_no_code_no_slots h = true))
    by exact (proj1 H).
  apply expr_handler_ind.
  - intros x kind d body IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hd Hb]. rewrite (IH Hb), andb_true_r.
    destruct d; try discriminate Hd; reflexivity.
  - intros bindings body _ IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin.
    specialize (Hbs b Hin).
    destruct b as [ | | s [ | | t fields | | | ]]; try discriminate Hbs.
    reflexivity.
  - intros k h body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. exact H.
  - intros ac H. exact H.
  - intros scrutinee arms H. exact H.
  - intros H. exact H.
  - intros params metadata handler IH H. exact (IH H).
Qed.

(* [rebuild_refines], of a unit with closures without value slots, and
   with static sets of closures without value slots, but without code. *)
Lemma rebuild_refines_checked (ex : bool)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  (∀ f : field, is_local f = false) ∨ projection_free e = true ->
  in_flow_fragment e = true ->
  checked ex e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  program_refines_observably e
    (rebuild_unit_from_tables (traverse_unit flag u e) T
       preserve_direct_calls e)
  ∧ (ex = true ->
     program_refines e
       (rebuild_unit_from_tables (traverse_unit flag u e) T
          preserve_direct_calls e)).
Proof.
  intros Hlocal Hslots Hfrag Hcf Hwf Hexits Hcheck Hcheck_exact.
  rename preserve_direct_calls into b.
  pose proof (check_closed_sound is_local _ _ Hcheck) as Hc.
  pose proof (check_field_used_exact_sound is_local _ _ Hcheck_exact)
    as Hexact.
  destruct (traverse_unit_facts flag u e (∀ f : field, is_local f = false)
              Hfrag Hslots)
    as (noted & deps & cds & He & Hconsts & Himported & Hunit & Hdeps & Hnd
        & Hkeys).
  pose proof (traverse_unit_scrutinees ex flag u e Hcf) as Hscr.
  set (F := traverse_unit flag u e) in *.
  set (G := graph_of_facts F) in *.
  set (sol := solution_of_tables T) in *.
  (* What FlambdaReaperFlowProofs.flow_sound derives of the graph. *)
  pose proof (static_names_edges G noted deps cds (unit_kept e) (unit_codes e)
                (static_names symbols_of_binding e)
                (∀ f : field, is_local f = false)) as [Hnames _].
  destruct (Hnames e None (unit_conts u) He) as [Hsym_names Hcode_names].
  assert (Hcds : ∀ (id : code_id) (cd : code_dep),
            lookup id cds = Some cd -> known_witness_edges G id cd).
  { intros id cd Hl.
    assert (Hid : id ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst cds (id, cd)).
      exact (lookup_in _ _ _ Hl). }
    destruct (Hcode_names id Hid) as (c & _ & Hcode).
    destruct c as [params mc ret exn metadata body].
    destruct Hcode as (cd' & Hin' & Hw & _).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hl. injection Hl as <-.
    exact Hw. }
  assert (Hunit_syms : ∀ s : symbol,
            among s (static_names symbols_of_binding e) = true ->
            may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs. apply among_In in Hs.
    destruct (Hsym_names s Hs) as (const & Hb).
    destruct const as [ | slot syms decls vs | t fields | | | ];
      simpl in Hb; try contradiction.
    - destruct Hb as (Hslot & Hdep & _). unfold is_bound in Hslot.
      destruct (lookup slot decls) as [id | ] eqn:Hl; [ | discriminate Hslot ].
      exact (called_alloc is_local G sol cds Hc Hcds _ _
               (Hdeps _ (Hdep id eq_refl))).
    - destruct Hb as (_ & His & _).
      exact (alloc_self is_local G sol Hc _ _ _ His
               (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))). }
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs.
    destruct (among s (static_names symbols_of_binding e)) eqn:Hunit_s;
      [ exact (Hunit_syms s Hunit_s) | ].
    apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _ (Himported s Hs Hunit_s)). }
  (* The continuations that the unit names are its own. *)
  assert (Hfree : ∀ k : continuation,
            k ∈ free_continuations_of_expr e -> unit_cont u k).
  { assert (Hnames_in : names_in_scope (unit_return u) (unit_exn u) e = true).
    { unfold well_formed_unit in Hwf.
      destruct (names_in_scope (unit_return u) (unit_exn u) e);
        [ reflexivity | ].
      rewrite andb_false_r in Hwf. discriminate Hwf. }
    unfold names_in_scope, names_among in Hnames_in.
    apply andb_true_iff in Hnames_in as [_ Hks].
    intros k Hk.
    pose proof (proj1 (all_among _ _) Hks k (proj2 (among_In _ _) Hk)) as Hin.
    apply among_In in Hin. destruct Hin as [<- | [<- | []]];
      [ left | right ]; reflexivity. }
  unfold program_refines, rebuild_unit_from_tables, rebuild_unit.
  eapply (unit_refines is_local G sol noted deps cds (unit_kept e)
            (unit_codes e) (static_names symbols_of_binding e) u F b);
    eassumption || reflexivity || exact (conj Hexits (conj Hscr Hcf)).
Qed.

(* The check of the proof, in each mode, is that of the statement. *)
Lemma checked_modes :
  (∀ e : expr,
     checked true e = no_code_no_slots e ∧ checked false e = no_call e)
  ∧ (∀ h : continuation_handler,
       handler_checked true h = handler_no_code_no_slots h
       ∧ handler_checked false h = handler_no_call h).
Proof.
  apply expr_handler_ind; simpl.
  - intros x kind d body [IH1 IH2]. rewrite IH1, IH2. auto.
  - intros bindings body _ [IH1 IH2]. rewrite IH1, IH2. auto.
  - intros k h body [IHh1 IHh2] [IH1 IH2]. rewrite IHh1, IHh2, IH1, IH2.
    auto.
  - intros handlers body IHh [IH1 IH2]. rewrite IH1, IH2.
    assert (Hl : ∀ l : list (continuation * continuation_handler),
              (∀ (k : continuation) (h : continuation_handler),
                 (k, h) ∈ l ->
                 handler_checked true h = handler_no_code_no_slots h
                 ∧ handler_checked false h = handler_no_call h) ->
              forallb (fun '(_, h) => handler_checked true h) l
              = forallb (fun '(_, h) => handler_no_code_no_slots h) l
              ∧ forallb (fun '(_, h) => handler_checked false h) l
                = forallb (fun '(_, h) => handler_no_call h) l).
    { induction l as [ | [k h] l IHl]; intros H; [ auto | ].
      simpl. destruct (H k h (or_introl eq_refl)) as [-> ->].
      destruct (IHl (fun k' h' Hin => H k' h' (or_intror Hin)))
        as [-> ->].
      auto. }
    destruct (Hl handlers IHh) as [-> ->]. auto.
  - auto.
  - auto.
  - auto.
  - auto.
  - intros params metadata handler IH. exact IH.
Qed.

Theorem no_code_no_call : no_code_no_call_statement.
Proof. exact (proj1 no_code_no_call_all). Qed.

Theorem no_code_no_slots_no_code : no_code_no_slots_no_code_statement.
Proof.
  enough (H : (∀ e : expr, no_code_no_slots e = true -> no_code e = true)
              ∧ (∀ h : continuation_handler,
                   handler_no_code_no_slots h = true ->
                   handler_no_code h = true))
    by exact (proj1 H).
  apply expr_handler_ind.
  - intros x kind d body IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hd Hb]. rewrite (IH Hb), andb_true_r.
    destruct d; simpl in Hd |- *; try discriminate Hd; reflexivity.
  - intros bindings body _ IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hbs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin.
    specialize (Hbs b Hin).
    destruct b as [ | | s [ | slot syms decls [ | ] | | | | ]];
      simpl in Hbs |- *; try discriminate Hbs; try reflexivity; exact Hbs.
  - intros k h body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hh Hb]. rewrite (IHh Hh), (IH Hb).
    reflexivity.
  - intros handlers body IHh IH H. simpl in H |- *.
    apply andb_true_iff in H as [Hhs Hb]. rewrite (IH Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin (Hhs _ Hin)).
  - intros ap H. exact H.
  - intros ac H. exact H.
  - intros scrutinee arms H. exact H.
  - intros H. exact H.
  - intros params metadata handler IH H. exact (IH H).
Qed.

Theorem rebuild_refines : rebuild_refines_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hcf Hwf Hexits Hcheck
    Hcheck_exact.
  refine (proj2 (rebuild_refines_checked true flag is_local u e T b Hlocal
                   Hslots Hfrag _ Hwf Hexits Hcheck Hcheck_exact) eq_refl).
  rewrite (proj1 (proj1 checked_modes e)).
  exact (closure_free_no_code_no_slots e Hcf).
Qed.

Theorem example : example_statement.
Proof.
  split; [ intros f Hf; discriminate Hf | ].
  split; [ intros f; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { intros T Hin. vm_compute in Hin.
    repeat (destruct Hin as [<- | Hin]; [ vm_compute; reflexivity | ]).
    destruct Hin. }
  split; [ vm_compute; reflexivity | ].
  split; vm_compute; reflexivity.
Qed.

Theorem exits_one_value_needed : exits_one_value_needed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State TwoValuesExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive
         [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 5)]
         empty_heap)).
  { exists 2. vm_compute. reflexivity. }
  assert (Htarget :
    behaves (State TwoValuesExample.rebuilt [] [] empty_static_env empty_heap)
      (Exits 1%positive
         [Value_const (Tagged_immediate 1); Value_const (Poison Value)]
         empty_heap)).
  { exists 1. vm_compute. reflexivity. }
  pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hsource) Htarget) as Heq.
  inversion Heq.
Qed.

Theorem field_used_exact_needed : field_used_exact_needed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State UnreadFieldExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive [Value_const (Tagged_immediate 0)] empty_heap)).
  { exists 5. vm_compute. reflexivity. }
  assert (Htarget :
    behaves
      (State UnreadFieldExample.rebuilt [] [] empty_static_env empty_heap)
      Goes_wrong).
  { exists 1. vm_compute. reflexivity. }
  pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hsource) Htarget) as Heq.
  discriminate Heq.
Qed.

Theorem closure_exit_counterexample : closure_exit_counterexample_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State ClosureExitExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive
         [Value_closure_of_set 70%positive [(70%positive, 50%positive)]
            [(60%positive, Value_const (Tagged_immediate 5))]]
         empty_heap)).
  { exists 4. vm_compute. reflexivity. }
  assert (Htarget :
    behaves
      (State ClosureExitExample.rebuilt [] [] empty_static_env empty_heap)
      (Exits 1%positive
         [Value_closure_of_set 70%positive [(70%positive, 50%positive)] []]
         empty_heap)).
  { exists 3. vm_compute. reflexivity. }
  pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hsource) Htarget) as Heq.
  inversion Heq.
Qed.

Theorem wrapper_kinds_counterexample :
  wrapper_kinds_counterexample_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  intros [Hexits _].
  assert (Hsource :
    behaves
      (State WrapperKindsExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive [Value_const (Tagged_immediate 0)] empty_heap)).
  { exists 6. vm_compute. reflexivity. }
  assert (Htarget :
    behaves
      (State WrapperKindsExample.rebuilt [] [] empty_static_env empty_heap)
      Goes_wrong).
  { exists 6. vm_compute. reflexivity. }
  pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hsource) Htarget) as Heq.
  discriminate Heq.
Qed.

(* ======================================================================== *)
(* Refinement up to observation                                             *)
(* ======================================================================== *)

Theorem refines_observably_of_refines :
  refines_observably_of_refines_statement.
Proof.
  intros s t [Hexits Hends]. split; [ | split ].
  - intros k vs hp Hs. exists vs.
    split; [ exact (Hexits k vs hp Hs) | apply values_observed_refl ].
  - intros k vs' hp Ht.
    destruct (Hends (Exits k vs' hp) ltac:(discriminate) Ht) as [Hs | Hw].
    + left. exists vs'. split; [ exact Hs | apply values_observed_refl ].
    + right. exact Hw.
  - intros Ht.
    destruct (Hends Goes_wrong ltac:(discriminate) Ht) as [Hs | Hw];
      assumption.
Qed.

Theorem rebuild_refines_observably : rebuild_refines_observably_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hnc _ Hwf Hexits Hcheck
    Hcheck_exact.
  refine (proj1 (rebuild_refines_checked false flag is_local u e T b Hlocal
                   Hslots Hfrag _ Hwf Hexits Hcheck Hcheck_exact)).
  rewrite (proj2 (proj1 checked_modes e)). exact Hnc.
Qed.

Theorem closure_exit_observed : closure_exit_observed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  assert (Hs :
    behaves
      (State ClosureExitExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive
         [Value_closure_of_set 70%positive [(70%positive, 50%positive)]
            [(60%positive, Value_const (Tagged_immediate 5))]]
         empty_heap))
    by (exists 4; vm_compute; reflexivity).
  assert (Ht :
    behaves
      (State ClosureExitExample.rebuilt [] [] empty_static_env empty_heap)
      (Exits 1%positive
         [Value_closure_of_set 70%positive [(70%positive, 50%positive)] []]
         empty_heap))
    by (exists 3; vm_compute; reflexivity).
  assert (Hobs :
    values_related observed_equal
      [Value_closure_of_set 70%positive [(70%positive, 50%positive)]
         [(60%positive, Value_const (Tagged_immediate 5))]]
      [Value_closure_of_set 70%positive [(70%positive, 50%positive)] []])
    by (cbn; repeat split).
  unfold program_refines_observably, refines_observably.
  split; [ | split ].
  - intros k vs hp Hb. pose proof (behaviour_unique _ _ _ Hs Hb) as Heq.
    injection Heq as <- <- <-. eexists. split; [ exact Ht | exact Hobs ].
  - intros k vs' hp Hb. pose proof (behaviour_unique _ _ _ Ht Hb) as Heq.
    injection Heq as <- <- <-. left. eexists. split; [ exact Hs | exact Hobs ].
  - intros Hw. pose proof (behaviour_unique _ _ _ Ht Hw) as Heq.
    discriminate Heq.
Qed.

Theorem results_kinded_needed : results_kinded_needed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  intros (_ & _ & Hwrong).
  assert (Hs :
    behaves
      (State WrapperKindsExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive [Value_const (Tagged_immediate 0)] empty_heap))
    by (exists 6; vm_compute; reflexivity).
  assert (Ht :
    behaves
      (State WrapperKindsExample.rebuilt [] [] empty_static_env empty_heap)
      Goes_wrong)
    by (exists 6; vm_compute; reflexivity).
  pose proof (behaviour_unique _ _ _ Hs (Hwrong Ht)) as Heq.
  discriminate Heq.
Qed.

Theorem closures_example : closures_example_statement.
Proof.
  do 8 (split; [ vm_compute; reflexivity | ]).
  split; exists 10; vm_compute; reflexivity.
Qed.

Theorem value_slots_example : value_slots_example_statement.
Proof.
  do 8 (split; [ vm_compute; reflexivity | ]).
  split; [ exists 12; vm_compute; reflexivity | ].
  split; [ exists 12; vm_compute; reflexivity | ].
  split.
  - unfold ValueSlotsExample.source_result, ValueSlotsExample.rebuilt_result,
      ValueSlotsExample.result.
    cbn. repeat split.
  - intros H. vm_compute in H. discriminate H.
Qed.

Theorem code_example : code_example_statement.
Proof.
  do 7 (split; [ vm_compute; reflexivity | ]).
  vm_compute. reflexivity.
Qed.
