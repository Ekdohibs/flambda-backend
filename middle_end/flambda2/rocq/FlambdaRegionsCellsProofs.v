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

(* Proofs for the cells of the machine of regions
   (FlambdaRegionsSpec.v, FlambdaRegionsInvariantSpec.v): the frames
   and their marks; the tests of what a value holds, of locations;
   what the fourth part of the invariant, [locations_all_bound], says,
   and that a step keeps it. Nothing here needs review: no statement
   of a Spec file is proved here, and FlambdaRegionsInvariantProofs.v
   uses what is. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaHeapProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaClosuresProofs FlambdaHeapMachineSpec
  FlambdaLocationsSpec FlambdaLocationsPrimitiveSpec
  FlambdaLocationsMachineSpec FlambdaLocationsMachineProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs.
From OxCamlFlambda2 Require FlambdaHeapMachineProofs.
Import ListNotations.

(* ======================================================================== *)
(* The frames and their marks                                               *)
(* ======================================================================== *)

Lemma on_stack_cons (m : mark) (f : frame) (frames : list frame) :
  on_stack m (f :: frames) = Pos.eqb (frame_mark f) m || on_stack m frames.
Proof. reflexivity. Qed.

Lemma on_stack_marks (m : mark) (frames : list frame) :
  on_stack m frames = true <-> In m (map frame_mark frames).
Proof.
  unfold on_stack. rewrite existsb_exists, in_map_iff.
  split; intros (f & H1 & H2); exists f.
  - split; [ apply Pos.eqb_eq; exact H2 | exact H1 ].
  - split; [ exact H2 | apply Pos.eqb_eq; exact H1 ].
Qed.

(* Whether a mark is on the stack is a matter of the marks of the
   frames, not of what they bind. *)
Lemma on_stack_same_marks (m : mark) (frames frames' : list frame) :
  map frame_mark frames = map frame_mark frames' ->
  on_stack m frames = on_stack m frames'.
Proof.
  intros Hmarks.
  destruct (on_stack m frames) eqn:H, (on_stack m frames') eqn:H';
    try reflexivity.
  - apply on_stack_marks in H. rewrite Hmarks in H.
    apply on_stack_marks in H. congruence.
  - apply on_stack_marks in H'. rewrite <- Hmarks in H'.
    apply on_stack_marks in H'. congruence.
Qed.

(* The frames from that of a mark down are the end of the stack, and
   no frame above them has the mark. *)
Lemma frames_from_app (m : mark) (frames fr : list frame) :
  frames_from m frames = Some fr ->
  ∃ above : list frame,
    frames = above ++ fr
    ∧ on_stack m above = false
    ∧ on_stack m fr = true.
Proof.
  revert fr.
  induction frames as [ | f below IH]; intros fr; simpl; [ discriminate | ].
  destruct (Pos.eqb (frame_mark f) m) eqn:Hm.
  - intros [= <-]. exists []. split; [ reflexivity | ].
    split; [ reflexivity | ]. rewrite on_stack_cons, Hm. reflexivity.
  - intros H. destruct (IH fr H) as (above & -> & Habove & Hfr).
    exists (f :: above). split; [ reflexivity | ].
    split; [ rewrite on_stack_cons, Hm; exact Habove | exact Hfr ].
Qed.

Lemma frames_from_on_stack (m : mark) (frames fr : list frame) :
  frames_from m frames = Some fr -> on_stack m frames = true.
Proof.
  intros H. destruct (frames_from_app m frames fr H) as (above & -> & _ & Hfr).
  rewrite on_stack_app, Hfr. apply orb_true_r.
Qed.

Lemma on_stack_frames_from (m : mark) (frames : list frame) :
  on_stack m frames = true ->
  ∃ fr : list frame, frames_from m frames = Some fr.
Proof.
  induction frames as [ | f below IH];
    [ intros H; unfold on_stack in H; discriminate | ].
  rewrite on_stack_cons. simpl. destruct (Pos.eqb (frame_mark f) m).
  - intros _. exists (f :: below). reflexivity.
  - exact IH.
Qed.

Lemma frames_from_none (m : mark) (frames : list frame) :
  on_stack m frames = false -> frames_from m frames = None.
Proof.
  intros H. destruct (frames_from m frames) as [fr | ] eqn:Hfr;
    [ | reflexivity ].
  rewrite (frames_from_on_stack m frames fr Hfr) in H. discriminate.
Qed.

(* Under frames none of which has the mark. *)
Lemma frames_from_under (m : mark) (above frames : list frame) :
  on_stack m above = false ->
  frames_from m (above ++ frames) = frames_from m frames.
Proof.
  induction above as [ | f above IH]; [ reflexivity | ].
  rewrite on_stack_cons. intros H. apply orb_false_iff in H as [Hf Habove].
  simpl. rewrite Hf. exact (IH Habove).
Qed.

Lemma nodup_app_apart {A : Type} (l l' : list A) (a : A) :
  NoDup (l ++ l') -> In a l -> In a l' -> False.
Proof.
  induction l as [ | b l IH]; simpl; intros H; [ intros [] | ].
  inversion H as [ | b0 l0 Hnotin Hrest]; subst.
  intros [<- | Hin] Hin'.
  - apply Hnotin, in_or_app. right. exact Hin'.
  - exact (IH Hrest Hin Hin').
Qed.

(* Frames no two of which have the same mark: the frames below some
   of them are such frames, and none of their marks is above. *)
Lemma distinct_marks_app (above below : list frame) :
  distinct (map frame_mark (above ++ below)) = true ->
  distinct (map frame_mark below) = true
  ∧ ∀ m : mark, on_stack m below = true -> on_stack m above = false.
Proof.
  rewrite map_app. intros H. apply distinct_NoDup in H. split.
  - apply distinct_NoDup. exact (NoDup_app_remove_l _ _ H).
  - intros m Hbelow. apply on_stack_marks in Hbelow.
    destruct (on_stack m above) eqn:Habove; [ | reflexivity ].
    apply on_stack_marks in Habove.
    destruct (nodup_app_apart _ _ m H Habove Hbelow).
Qed.

(* So the frames from a mark down are the same under more frames. *)
Lemma frames_from_below (m : mark) (above below fr : list frame) :
  distinct (map frame_mark (above ++ below)) = true ->
  frames_from m below = Some fr ->
  frames_from m (above ++ below) = Some fr.
Proof.
  intros Hdistinct Hfr.
  destruct (distinct_marks_app above below Hdistinct) as [_ Habove].
  rewrite (frames_from_under m above below); [ exact Hfr | ].
  exact (Habove m (frames_from_on_stack m below fr Hfr)).
Qed.

(* A frame with a mark that no frame has, on top. *)
Lemma distinct_marks_cons (f : frame) (frames : list frame) :
  on_stack (frame_mark f) frames = false ->
  distinct (map frame_mark frames) = true ->
  distinct (map frame_mark (f :: frames)) = true.
Proof.
  intros Hf Hframes. simpl. rewrite Hframes, andb_true_r.
  destruct (among (frame_mark f) (map frame_mark frames)) eqn:Hamong;
    [ | reflexivity ].
  apply among_In, on_stack_marks in Hamong. congruence.
Qed.

(* ======================================================================== *)
(* What a value holds, of locations                                         *)
(* ======================================================================== *)

(* The tests of what a value holds, for another set of locations:
   one that has every location of the first set that the value
   names. *)
Lemma value_holds_cells_in (ok ok' : location -> bool) (v : value) :
  (∀ l : location,
    In l (locations_of_value v) -> ok l = true -> ok' l = true) ->
  value_holds_cells ok v = true ->
  value_holds_cells ok' v = true.
Proof.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    simpl; intros Hok; auto; try (exact (Hok _ (or_introl eq_refl))).
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    apply (IH (vslot, w) Hin); [ | exact (forallb_in _ _ (vslot, w) H Hin) ].
    intros l Hl. apply Hok, in_flat_map. exists (vslot, w).
    split; assumption.
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    apply (IH (vslot, w) Hin); [ | exact (forallb_in _ _ (vslot, w) H Hin) ].
    intros l Hl. apply Hok, in_flat_map. exists (vslot, w).
    split; assumption.
  - intros H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH.
    apply (IH w Hin); [ | exact (forallb_in _ _ w H Hin) ].
    intros l Hl. apply Hok, in_flat_map. exists w. split; assumption.
  - intros H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH.
    apply (IH w Hin); [ | exact (forallb_in _ _ w H Hin) ].
    intros l Hl. apply Hok, in_flat_map. exists w. split; assumption.
  - intros H. apply andb_true_iff in H as [Hcallee Hargs].
    apply andb_true_iff. split.
    + destruct callee; try reflexivity;
        (apply IHcallee; [ | exact Hcallee ]);
        intros l Hl; apply Hok, in_or_app; left; exact Hl.
    + apply forallb_forall. intros arg Hin.
      rewrite Forall_forall in IHargs.
      pose proof (forallb_in _ _ arg Hargs Hin) as Harg.
      cbv beta in Harg.
      assert (Hok' :
        ∀ l : location,
          In l (locations_of_value arg) -> ok l = true -> ok' l = true).
      { intros l Hl. apply Hok, in_or_app. right. apply in_flat_map.
        exists arg. split; assumption. }
      destruct arg; try reflexivity; exact (IHargs _ Hin Hok' Harg).
Qed.

Lemma value_holds_cells_kept (ok : location -> bool) (v : value) :
  value_holds_cells ok v = true -> value_kept_cells ok v = true.
Proof. destruct v; simpl; auto. Qed.

Lemma value_placed_cells_kept (near ok : location -> bool) (v : value) :
  value_placed_cells near ok v = true -> value_kept_cells ok v = true.
Proof. destruct v; simpl; auto. Qed.

Lemma value_kept_cells_in (ok ok' : location -> bool) (v : value) :
  (∀ l : location,
    In l (locations_of_value v) -> ok l = true -> ok' l = true) ->
  value_kept_cells ok v = true ->
  value_kept_cells ok' v = true.
Proof.
  intros Hok.
  destruct v;
    first
      [ exact (value_holds_cells_in ok ok' _ Hok) | intros _; reflexivity ].
Qed.

Lemma value_placed_cells_in (near near' ok ok' : location -> bool)
    (v : value) :
  (∀ l : location,
    In l (locations_of_value v) -> near l = true -> near' l = true) ->
  (∀ l : location,
    In l (locations_of_value v) -> ok l = true -> ok' l = true) ->
  value_placed_cells near ok v = true ->
  value_placed_cells near' ok' v = true.
Proof.
  intros Hnear Hok.
  destruct v;
    first
      [ exact (Hnear _ (or_introl eq_refl))
      | exact (value_holds_cells_in ok ok' _ Hok) ].
Qed.

(* A value that is not a location and that a variable may hold is one
   that anything may hold. *)
Lemma value_kept_cells_holds (ok : location -> bool) (v : value) :
  (∀ l : location, v = Value_location l -> ok l = true) ->
  value_kept_cells ok v = true ->
  value_holds_cells ok v = true.
Proof.
  intros Hl. destruct v; simpl; auto.
Qed.

Lemma value_kept_cells_placed (near ok : location -> bool) (v : value) :
  (∀ l : location, v = Value_location l -> near l = true) ->
  value_kept_cells ok v = true ->
  value_placed_cells near ok v = true.
Proof.
  intros Hl. destruct v; simpl; auto.
Qed.

Lemma value_holds_cells_parts (ok : location -> bool) (v : value) :
  value_holds_cells ok v = true ->
  forallb (value_holds_cells ok) (value_parts v) = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs; auto.
Qed.

Lemma value_holds_cells_made (ok : location -> bool) (v : value) :
  is_made v = true ->
  forallb (value_holds_cells ok) (value_parts v) = true ->
  value_holds_cells ok v = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs; try discriminate; auto.
Qed.

Lemma kept_cells_parts (ok : location -> bool) (v w : value) :
  value_kept_cells ok v = true ->
  In w (value_parts v) ->
  value_kept_cells ok w = true.
Proof.
  intros Hv Hin.
  destruct v; simpl in Hin; try contradiction;
    apply value_holds_cells_kept;
    exact (forallb_in _ _ w (value_holds_cells_parts ok _ Hv) Hin).
Qed.

Lemma eval_simple_kept_cells (ok : location -> bool) (vars : env)
    (sim : simple) (v : value) :
  forallb (fun '(_, v) => value_kept_cells ok v) vars = true ->
  eval_simple vars sim = Some v ->
  value_kept_cells ok v = true.
Proof.
  intros Hvars. destruct sim as [x | sym | c]; simpl.
  - exact (lookup_forallb (value_kept_cells ok) x v vars Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

(* The two lists of the values that what a symbol is bound to holds
   directly are one list. *)
Lemma values_of_static_values (allocated : statically_allocated_value) :
  values_of_static allocated = allocated_values allocated.
Proof. destruct allocated; reflexivity. Qed.

(* [cont_kept_cells] has no test of the handlers, as
   FlambdaRegionsInvariantSpec.cont_kept. *)
Lemma conts_kept_cells_split (ok : location -> bool) (conts : cont_env) :
  forallb (fun '(_, c) => cont_kept_cells ok c) conts = true
  <-> conts_values_all (fun v => value_kept_cells ok v = true) conts.
Proof.
  assert (Hsplit :
    forallb (fun '(_, c) => cont_kept_cells ok c) conts = true
    <-> conts_terms_ok (fun _ => true) conts
        ∧ conts_values_all (fun v => value_kept_cells ok v = true) conts).
  { apply
      (conts_split_all (fun _ => true) (value_kept_cells ok)
         (cont_kept_cells ok));
      try reflexivity.
    intros h vars conts' recursive. simpl.
    rewrite forallb_pairs_true, andb_true_r. reflexivity. }
  destruct Hsplit as [H1 H2]. split.
  - intros H. exact (proj2 (H1 H)).
  - intros H. exact (H2 (conj (conts_terms_ok_true conts) H)).
Qed.

(* ======================================================================== *)
(* The locations that a state holds                                         *)
(* ======================================================================== *)

Lemma locations_of_parts (v w : value) (l : location) :
  In w (value_parts v) ->
  In l (locations_of_value w) ->
  In l (locations_of_value v).
Proof.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c
       | l0 | callee args args_arity];
    simpl; try contradiction.
  - intros Hw Hl. apply in_map_iff in Hw as ([vslot w'] & <- & Hin).
    apply in_flat_map. exists (vslot, w'). split; assumption.
  - intros Hw Hl. apply in_map_iff in Hw as ([vslot w'] & <- & Hin).
    apply in_flat_map. exists (vslot, w'). split; assumption.
  - intros Hw Hl. apply in_flat_map. exists w. split; assumption.
  - intros Hw Hl. apply in_flat_map. exists w. split; assumption.
  - intros [<- | []] [].
Qed.

Lemma value_closed_parts (hp : heap) (v w : value) :
  value_closed hp v -> In w (value_parts v) -> value_closed hp w.
Proof.
  intros Hv Hin l Hl. exact (Hv l (locations_of_parts v w l Hin Hl)).
Qed.

Lemma value_closed_mono (hp hp' : heap) (v : value) :
  (∀ l : location, is_bound l hp = true -> is_bound l hp' = true) ->
  value_closed hp v ->
  value_closed hp' v.
Proof. intros Hkept Hv l Hl. exact (Hkept l (Hv l Hl)). Qed.

Lemma locations_bound_app (hp : heap) (ls ls' : list location) :
  locations_bound hp (ls ++ ls')
  <-> locations_bound hp ls ∧ locations_bound hp ls'.
Proof.
  unfold locations_bound. split.
  - intros H.
    split; intros l Hl; apply H, in_or_app; [ left | right ]; exact Hl.
  - intros [H1 H2] l Hl.
    apply in_app_or in Hl as [Hl | Hl]; [ exact (H1 l Hl) | exact (H2 l Hl) ].
Qed.

Lemma locations_bound_values (hp : heap) (vs : list value) :
  locations_bound hp (flat_map locations_of_value vs)
  <-> Forall (value_closed hp) vs.
Proof.
  unfold locations_bound, value_closed. rewrite Forall_forall. split.
  - intros H v Hin l Hl. apply H, in_flat_map. exists v. split; assumption.
  - intros H l Hl. apply in_flat_map in Hl as (v & Hin & Hl).
    exact (H v Hin l Hl).
Qed.

Lemma value_closed_partial (hp : heap) (callee : value) (args : list value)
    (args_arity : arity) :
  value_closed hp (Value_partial_application callee args args_arity)
  <-> value_closed hp callee ∧ Forall (value_closed hp) args.
Proof.
  rewrite <- locations_bound_values.
  exact (locations_bound_app hp (locations_of_value callee) _).
Qed.

Lemma env_closed (hp : heap) (vars : env) :
  env_values_all (value_closed hp) vars
  <-> locations_bound hp (locations_of_env vars).
Proof.
  unfold env_values_all, locations_bound, locations_of_env.
  rewrite Forall_forall. split.
  - intros H l Hl. apply in_flat_map in Hl as (b & Hin & Hl).
    exact (H b Hin l Hl).
  - intros H b Hin l Hl. apply H, in_flat_map. exists b. split; assumption.
Qed.

Lemma cont_closed (hp : heap) (c : cont_value) :
  cont_values_all (value_closed hp) c
  <-> locations_bound hp (locations_of_cont_value c).
Proof.
  induction c
    as [h vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_deep_ind;
    cbn [cont_values_all locations_of_cont_value].
  - assert (Hconts :
      fold_right
        (fun binding rest =>
           cont_values_all (value_closed hp) (snd binding) ∧ rest)
        True conts
      <-> locations_bound hp
            (flat_map (fun binding => locations_of_cont_value (snd binding))
               conts)).
    { induction IH as [ | kc conts Hkc _ IHconts]; simpl.
      - split; [ intros _ l [] | intros _; exact I ].
      - rewrite locations_bound_app, Hkc, IHconts. reflexivity. }
    rewrite locations_bound_app, env_closed, Hconts. reflexivity.
  - split; [ intros _ l [] | intros _; exact I ].
  - rewrite locations_bound_app, locations_bound_values, IH. tauto.
  - rewrite !locations_bound_app, locations_bound_values, IHreturn, IHexn.
    reflexivity.
Qed.

Lemma conts_closed (hp : heap) (conts : cont_env) :
  conts_values_all (value_closed hp) conts
  <-> locations_bound hp (locations_of_cont_env conts).
Proof.
  unfold conts_values_all, locations_of_cont_env.
  induction conts as [ | kc conts IH]; simpl.
  - split; [ intros _ l [] | intros _; exact I ].
  - rewrite locations_bound_app, cont_closed, IH. reflexivity.
Qed.

Lemma vals_closed (hp : heap) (s : state) :
  vals_ok (value_closed hp) s
  <-> locations_bound hp (locations_of_env (state_vars s))
      ∧ locations_bound hp (locations_of_cont_env (state_conts s)).
Proof. unfold vals_ok. rewrite env_closed, conts_closed. reflexivity. Qed.

(* Two properties of every value that a continuation holds. *)
Fixpoint cont_values_all_and (P Q : value -> Prop) (c : cont_value) :
  cont_values_all P c ->
  cont_values_all Q c ->
  cont_values_all (fun v => P v ∧ Q v) c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    cbn [cont_values_all]; [ | auto | | ].
  - intros [Hvars Hconts] [Hvars' Hconts']. split.
    + apply Forall_forall. intros binding Hin.
      exact (conj (proj1 (Forall_forall _ _) Hvars binding Hin)
               (proj1 (Forall_forall _ _) Hvars' binding Hin)).
    + induction conts as [ | [k c] conts IH]; cbn [fold_right]; [ exact I | ].
      destruct Hconts as [Hc Hrest]. destruct Hconts' as [Hc' Hrest'].
      split;
        [ exact (cont_values_all_and P Q c Hc Hc') | exact (IH Hrest Hrest') ].
  - intros [Hextra Hc] [Hextra' Hc'].
    split; [ | exact (cont_values_all_and P Q c Hc Hc') ].
    apply Forall_forall. intros v Hin.
    exact (conj (proj1 (Forall_forall _ _) Hextra v Hin)
             (proj1 (Forall_forall _ _) Hextra' v Hin)).
  - intros (Hremaining & Hreturn & Hexn) (Hremaining' & Hreturn' & Hexn').
    split; [ | split ].
    + apply Forall_forall. intros v Hin.
      exact (conj (proj1 (Forall_forall _ _) Hremaining v Hin)
               (proj1 (Forall_forall _ _) Hremaining' v Hin)).
    + exact (cont_values_all_and P Q return_to Hreturn Hreturn').
    + exact (cont_values_all_and P Q exn_to Hexn Hexn').
Qed.

Lemma conts_values_all_and (P Q : value -> Prop) (conts : cont_env) :
  conts_values_all P conts ->
  conts_values_all Q conts ->
  conts_values_all (fun v => P v ∧ Q v) conts.
Proof.
  unfold conts_values_all.
  induction conts as [ | [k c] conts IH]; cbn [fold_right]; [ auto | ].
  intros [Hc Hrest] [Hc' Hrest'].
  split; [ exact (cont_values_all_and P Q c Hc Hc') | exact (IH Hrest Hrest') ].
Qed.

Lemma env_values_all_and (P Q : value -> Prop) (vars : env) :
  env_values_all P vars ->
  env_values_all Q vars ->
  env_values_all (fun v => P v ∧ Q v) vars.
Proof.
  unfold env_values_all. intros H H'. apply Forall_forall. intros binding Hin.
  exact (conj (proj1 (Forall_forall _ _) H binding Hin)
           (proj1 (Forall_forall _ _) H' binding Hin)).
Qed.

(* A property of the values that a state holds, from another, where
   the heap binds every location that they name. *)
Lemma vals_ok_closed (P P' : value -> Prop) (hp : heap) (s : state) :
  (∀ v : value, value_closed hp v -> P v -> P' v) ->
  vals_ok (value_closed hp) s ->
  vals_ok P s ->
  vals_ok P' s.
Proof.
  intros Himpl [Hvars Hconts] [Hv Hc]. split.
  - exact
      (Forall_impl _ (fun b H => Himpl (snd b) (proj2 H) (proj1 H))
         (env_values_all_and _ _ _ Hv Hvars)).
  - exact
      (conts_values_all_impl _ P' _
         (fun v H => Himpl v (proj2 H) (proj1 H))
         (conts_values_all_and _ _ _ Hc Hconts)).
Qed.

Lemma eval_simple_closed (hp : heap) (vars : env) (sim : simple)
    (v : value) :
  env_values_all (value_closed hp) vars ->
  eval_simple vars sim = Some v ->
  value_closed hp v.
Proof.
  intros Hvars. destruct sim as [x | sym | c]; simpl.
  - intros Hlookup. unfold env_values_all in Hvars.
    rewrite Forall_forall in Hvars.
    exact (Hvars (x, v) (lookup_in _ _ _ Hlookup)).
  - intros [= <-] l [].
  - intros [= <-] l [].
Qed.

Lemma statics_closed (hp : heap) (statics : static_env) :
  locations_bound hp (locations_of_static_env statics)
  <-> ∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
        In (s, allocated) (static_values statics) ->
        In w (allocated_values allocated) ->
        value_closed hp w.
Proof.
  unfold locations_bound, locations_of_static_env, locations_of_static_value,
    value_closed.
  split.
  - intros H s allocated w Hin Hw l Hl. apply H, in_flat_map.
    exists (s, allocated). split; [ exact Hin | ]. cbn [snd].
    apply in_flat_map. exists w. rewrite values_of_static_values.
    split; assumption.
  - intros H l Hl. apply in_flat_map in Hl as ([s allocated] & Hin & Hl).
    cbn [snd] in Hl. apply in_flat_map in Hl as (w & Hw & Hl).
    rewrite values_of_static_values in Hw. exact (H s allocated w Hin Hw l Hl).
Qed.

Lemma live_statics_closed (hp : heap) (rs : region_state) :
  locations_bound hp (locations_of_static_env (live_statics rs))
  <-> (∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
        In (s, allocated) (live_locals (rs_frames rs)) ->
        In w (allocated_values allocated) ->
        value_closed hp w)
      ∧ (∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
          In (s, allocated) (static_values (state_statics (rs_state rs))) ->
          In w (allocated_values allocated) ->
          value_closed hp w).
Proof.
  rewrite statics_closed. unfold live_statics. cbn [static_values]. split.
  - intros H. split; intros s allocated w Hin Hw.
    + apply (H s allocated w); [ | exact Hw ]. apply in_or_app. left.
      exact Hin.
    + apply (H s allocated w); [ | exact Hw ]. apply in_or_app. right.
      exact Hin.
  - intros [H1 H2] s allocated w Hin.
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (H1 s allocated w Hin) | exact (H2 s allocated w Hin) ].
Qed.

Lemma heap_closed (hp hp0 : heap) :
  locations_bound hp (locations_of_heap hp0)
  <-> (∀ (l : location) (c : cell), In (l, c) hp0 -> is_bound l hp = true)
      ∧ heap_values_all (value_closed hp) hp0.
Proof.
  unfold locations_bound, locations_of_heap, locations_of_cell,
    heap_values_all, value_closed.
  split.
  - intros H. split.
    + intros l c Hin. apply H, in_flat_map. exists (l, c).
      split; [ exact Hin | left; reflexivity ].
    + intros l c v Hin Hv l0 Hl0. apply H, in_flat_map. exists (l, c).
      split; [ exact Hin | ]. right. cbn [snd]. apply in_flat_map.
      exists v. split; assumption.
  - intros [Hkeys Hvalues] l Hl.
    apply in_flat_map in Hl as ([l0 c] & Hin & Hl). cbn [fst snd] in Hl.
    destruct Hl as [<- | Hl].
    + exact (Hkeys l0 c Hin).
    + apply in_flat_map in Hl as (v & Hv & Hl).
      exact (Hvalues l0 c v Hin Hv l Hl).
Qed.

(* ======================================================================== *)
(* The fourth part of the invariant                                         *)
(* ======================================================================== *)

(* [locations_all_bound], as
   FlambdaLocationsMachineSpec.state_locations_bound of the state that
   [step] is given. *)
Definition bound (rs : region_state) : Prop :=
  state_locations_bound (with_statics (rs_state rs) (live_statics rs)).

Lemma locations_all_bound_iff (rs : region_state) :
  locations_all_bound rs = true <-> bound rs.
Proof.
  unfold locations_all_bound, bound, state_locations_bound, locations_bound.
  cbv zeta. rewrite forallb_forall. reflexivity.
Qed.

(* What it says: of the values of the variables and of the
   continuations, of what the frames and the static environment bind,
   of the heap and of the read record. *)
Lemma bound_parts (rs : region_state) :
  bound rs
  <-> let hp := state_heap (rs_state rs) in
      vals_ok (value_closed hp) (rs_state rs)
      ∧ (∀ (s : symbol) (allocated : statically_allocated_value)
           (w : value),
          In (s, allocated) (live_locals (rs_frames rs)) ->
          In w (allocated_values allocated) ->
          value_closed hp w)
      ∧ (∀ (s : symbol) (allocated : statically_allocated_value)
           (w : value),
          In (s, allocated) (static_values (state_statics (rs_state rs))) ->
          In w (allocated_values allocated) ->
          value_closed hp w)
      ∧ (∀ (l : location) (c : cell), In (l, c) hp -> is_bound l hp = true)
      ∧ heap_values_all (value_closed hp) hp
      ∧ locations_bound hp (locations_of_reads (state_reads (rs_state rs))).
Proof.
  cbv zeta. unfold bound, state_locations_bound, locations_of_state.
  cbn [with_statics state_vars state_conts state_statics state_heap
       state_reads].
  rewrite !locations_bound_app, live_statics_closed, heap_closed, vals_closed.
  tauto.
Qed.

Lemma eval_named_closed (rs : region_state) (d : named) (v : value) :
  bound rs ->
  eval_named (live_statics rs) (state_vars (rs_state rs)) d = Some v ->
  value_closed (state_heap (rs_state rs)) v.
Proof.
  intros Hb Heval l Hl.
  destruct (state_locations_bound_parts _ Hb) as (Hvars & _ & Hstatics & _).
  destruct (in_app_or _ _ _ (eval_named_locations _ _ d v l Heval Hl))
    as [Hin | Hin];
    [ exact (Hstatics l Hin) | exact (Hvars l Hin) ].
Qed.

(* What a [Let] runs: the heap after binds every location that it
   holds and every location of the value, and every location that the
   heap before bound; the read record is as it came. *)
Lemma exec_named_bound (rs : region_state) (d : named) (v : value)
    (hp' : heap) (rd' : read_record) :
  bound rs ->
  exec_named (live_statics rs) (state_heap (rs_state rs))
    (state_reads (rs_state rs)) (state_vars (rs_state rs)) d
  = Some (v, hp', rd') ->
  locations_bound hp' (locations_of_heap hp')
  ∧ value_closed hp' v
  ∧ (∀ l : location,
       is_bound l (state_heap (rs_state rs)) = true -> is_bound l hp' = true)
  ∧ rd' = state_reads (rs_state rs).
Proof.
  intros Hb Hexec.
  destruct (state_locations_bound_parts _ Hb)
    as (Hvars & _ & Hstatics & Hheap & _).
  rewrite FlambdaHeapMachineProofs.exec_named_heap_exec_named in Hexec.
  destruct
    (heap_exec_named (live_statics rs) (state_heap (rs_state rs))
       (state_vars (rs_state rs)) d)
    as [[v0 h0] | ] eqn:Hheap_exec;
    [ | discriminate ].
  cbn [option_map fst snd] in Hexec. injection Hexec as <- <- <-.
  destruct
    (heap_exec_named_keeps_locations_bound _ _ _ _ d v0 Hheap Hvars Hstatics
       Hheap_exec)
    as (H1 & H2 & H3).
  exact (conj H1 (conj H2 (conj H3 eq_refl))).
Qed.

(* ======================================================================== *)
(* A step keeps the fourth part                                             *)
(* ======================================================================== *)

Lemma begin_region_bound (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr) :
  bound rs ->
  begin_region rs x kind mark_kind body = Region_step rs' ->
  bound rs'.
Proof.
  intros Hb Hstep. rewrite (begin_region_step _ _ _ _ _ _ Hstep).
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as ([Hv Hc] & Hrest).
  apply bound_parts. cbv zeta. unfold bind_let.
  cbn [rs_state rs_frames state_vars state_conts state_heap state_reads
       state_statics].
  split; [ | exact Hrest ].
  split; [ apply Forall_cons; [ intros l [] | exact Hv ] | exact Hc ].
Qed.

Lemma end_region_bound (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  bound rs ->
  end_region rs x kind ended_by region body = Region_step rs' ->
  bound rs'.
Proof.
  intros Hb Hstep.
  destruct (end_region_step _ _ _ _ _ _ _ Hstep)
    as (m & above & f & below & counts & _ & Hframes & ->).
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as ([Hv Hc] & Hlocals & Hrest).
  apply bound_parts. cbv zeta. unfold bind_let.
  cbn [rs_state rs_frames state_vars state_conts state_heap state_reads
       state_statics].
  split;
    [ split; [ apply Forall_cons; [ intros l [] | exact Hv ] | exact Hc ] | ].
  split; [ | exact Hrest ].
  intros s allocated w Hin. apply (Hlocals s allocated w).
  rewrite Hframes, live_locals_app. apply in_or_app. right.
  change (live_locals (f :: below)) with (frame_locals f ++ live_locals below).
  apply in_or_app. right. exact Hin.
Qed.

Lemma local_allocation_bound (rs rs' : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  is_cell_allocation d && kind_eqb kind Value = false ->
  bound rs ->
  local_allocation rs x y kind d region body = Region_step rs' ->
  bound rs'.
Proof.
  intros Hnot Hb Hstep.
  destruct (local_allocation_step rs rs' x y kind d region body Hnot Hstep)
    as (v & sym & bindings & His & Heval & Hlocalise & _ & Hvars & Hconts & _
        & _ & _ & Hframes).
  destruct (local_allocation_memory rs rs' x y kind d region body Hnot Hstep)
    as (Hheap & Hreads & _).
  pose proof (eval_named_closed rs d v Hb Heval) as Hv.
  destruct (localise_makes _ _ _ _ Hlocalise) as (_ & _ & Hbound).
  assert (Hnew :
    ∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
      In (s, allocated) bindings ->
      In w (allocated_values allocated) ->
      value_closed (state_heap (rs_state rs)) w).
  { intros s allocated w Hin Hw.
    destruct (Hbound s allocated Hin) as (_ & _ & Hvalues & _).
    rewrite Hvalues in Hw. exact (value_closed_parts _ v w Hv Hw). }
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as ([Hv0 Hc0] & Hlocals & Hstatics & Hrest).
  apply bound_parts. cbv zeta. rewrite Hheap, Hreads.
  split; [ split | ].
  - rewrite Hvars. apply Forall_cons; [ intros l [] | ].
    apply Forall_cons; [ intros l [] | exact Hv0 ].
  - rewrite Hconts. exact Hc0.
  - destruct (rs_frames rs) as [ | [m mark_kind locals] frames].
    + destruct Hframes as [-> Hvalues']. rewrite Hvalues'.
      split; [ intros s allocated w [] | ]. split; [ | exact Hrest ].
      intros s allocated w Hin.
      apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hnew s allocated w Hin) | exact (Hstatics s allocated w Hin) ].
    + destruct Hframes as [-> ->]. split; [ | exact (conj Hstatics Hrest) ].
      intros s allocated w Hin.
      change (live_locals (Frame m mark_kind (bindings ++ locals) :: frames))
        with ((bindings ++ locals) ++ live_locals frames) in Hin.
      rewrite <- app_assoc in Hin.
      apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hnew s allocated w Hin) | exact (Hlocals s allocated w Hin) ].
Qed.

Lemma local_cell_allocation_bound (rs rs' : region_state) (x y : variable)
    (d : named) (region : option simple) (body : expr) :
  bound rs ->
  local_cell_allocation rs x y d region body = Region_step rs' ->
  bound rs'.
Proof.
  intros Hb Hstep.
  destruct (local_cell_allocation_step rs rs' x y d region body Hstep)
    as (l & hp' & rd' & _ & Hexec & Hstate & Hframes & _).
  destruct (exec_named_bound rs d _ hp' rd' Hb Hexec)
    as (Hheap' & Hl & Hkept & ->).
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as ([Hv0 Hc0] & Hlocals & Hstatics & _ & _ & Hrd).
  apply bound_parts. cbv zeta. rewrite Hstate, Hframes.
  cbn [state_vars state_conts state_statics state_heap state_reads].
  apply heap_closed in Hheap' as [Hkeys' Hcells'].
  split; [ split | ].
  - apply Forall_cons; [ exact Hl | ]. apply Forall_cons; [ exact Hl | ].
    exact (Forall_impl _ (fun b => value_closed_mono _ _ (snd b) Hkept) Hv0).
  - exact
      (conts_values_all_impl _ _ _ (fun v => value_closed_mono _ _ v Hkept)
         Hc0).
  - split.
    { intros s allocated w Hin Hw.
      exact (value_closed_mono _ _ w Hkept (Hlocals s allocated w Hin Hw)). }
    split.
    { intros s allocated w Hin Hw.
      exact (value_closed_mono _ _ w Hkept (Hstatics s allocated w Hin Hw)). }
    split; [ exact Hkeys' | ]. split; [ exact Hcells' | ].
    intros l0 Hl0. exact (Hkept l0 (Hrd l0 Hl0)).
Qed.

Lemma state_locations_bound_build (s : state) :
  locations_bound (state_heap s) (locations_of_env (state_vars s)) ->
  locations_bound (state_heap s) (locations_of_cont_env (state_conts s)) ->
  locations_bound (state_heap s)
    (locations_of_static_env (state_statics s)) ->
  locations_bound (state_heap s) (locations_of_heap (state_heap s)) ->
  locations_bound (state_heap s) (locations_of_reads (state_reads s)) ->
  state_locations_bound s.
Proof.
  intros H1 H2 H3 H4 H5 l Hl. unfold locations_of_state in Hl.
  rewrite !in_app_iff in Hl.
  destruct Hl as [Hl | [Hl | [Hl | [Hl | Hl]]]]; auto.
Qed.

Lemma plain_step_bound (rs rs' : region_state) :
  bound rs -> plain_step rs = Region_step rs' -> bound rs'.
Proof.
  intros Hb Hstep. destruct (plain_step_step _ _ Hstep) as (s' & Hs & ->).
  pose proof (step_keeps_locations_bound _ Hb) as H. rewrite Hs in H.
  cbn [result_locations_bound] in H. destruct H as [Hb' Hkept].
  destruct (state_locations_bound_parts _ Hb) as (_ & _ & Hstatics & _).
  destruct (state_locations_bound_parts _ Hb')
    as (Hvars' & Hconts' & _ & Hheap' & Hreads').
  apply state_locations_bound_build;
    cbn [rs_state rs_frames with_statics state_vars state_conts
         state_statics state_heap state_reads];
    try assumption.
  intros l Hl. apply Hkept. apply Hstatics. exact Hl.
Qed.

Lemma static_step_bound (rs rs' : region_state)
    (bindings : list static_binding) (body : expr) :
  state_expr (rs_state rs) = Let_static bindings body ->
  bound rs ->
  static_step rs = Region_step rs' ->
  bound rs'.
Proof.
  intros He Hb Hstep.
  destruct (static_step_step rs rs' bindings body He Hstep)
    as (statics' & Heval & ->).
  destruct (eval_static_bindings_parts _ _ _ _ Heval)
    as (new_code & new_values & _ & Hvalues & _ & Hnew).
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as ([Hv0 Hc0] & Hlocals & Hstatics & Hrest).
  apply bound_parts. cbv zeta.
  cbn [rs_state rs_frames state_vars state_conts state_statics state_heap
       state_reads].
  split; [ exact (conj Hv0 Hc0) | ]. split; [ exact Hlocals | ].
  split; [ | exact Hrest ].
  intros s allocated w Hin Hw. rewrite Hvalues in Hin.
  apply in_app_or in Hin as [Hin | Hin];
    [ | exact (Hstatics s allocated w Hin Hw) ].
  destruct (Hnew s allocated Hin) as (const & _ & Hconst).
  destruct (eval_static_const_makes _ const allocated Hconst) as [Hparts _].
  destruct (Hparts w Hw) as (sim & _ & Hsim).
  exact (eval_simple_closed _ _ sim w Hv0 Hsim).
Qed.

Lemma closed_fields (rs : region_state) :
  bound rs ->
  P_fields (value_closed (state_heap (rs_state rs))) (live_statics rs).
Proof.
  intros Hb.
  destruct (state_locations_bound_parts _ Hb) as (_ & _ & Hstatics & _).
  apply (reads_fields _ (live_statics rs)).
  - exact (value_closed_parts _).
  - intros s allocated _ Hlookup. split; [ | intros s' _ l [] ].
    intros w Hw.
    exact
      (proj1 (statics_closed _ _) Hstatics s allocated w
         (lookup_in _ _ _ Hlookup) Hw).
Qed.

Lemma region_apply_cont_bound (rs rs' : region_state) (ac : apply_cont) :
  state_expr (rs_state rs) = Apply_cont ac ->
  bound rs ->
  region_apply_cont rs ac = Region_step rs' ->
  bound rs'.
Proof.
  intros He Hb Hstep.
  pose proof Hb as Hparts. apply bound_parts in Hparts. cbv zeta in Hparts.
  destruct Hparts as (Hvals & Hrest).
  pose proof
    (region_apply_cont_values _ (value_closed_partial _) rs ac He
       (closed_fields rs Hb) Hvals
       (fun sim v _ => eval_simple_closed _ _ sim v (proj1 Hvals)))
    as Hvals'.
  rewrite Hstep in Hvals'. cbn [region_result_vals_ok] in Hvals'.
  revert Hstep. unfold region_apply_cont.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try exact (plain_step_bound rs rs' Hb).
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ];
    [ | discriminate ].
  destruct
    (kinds_eqb (map (region_kind_of_value rs) vs) (hm_params_kinds metadata));
    [ | discriminate ].
  destruct (bind_params params vs def_vars) as [handler_vars | ];
    [ | discriminate ].
  intros [= <-]. apply bound_parts. cbv zeta.
  cbn [rs_state rs_frames state_vars state_conts state_statics state_heap
       state_reads] in Hvals' |- *.
  exact (conj Hvals' Hrest).
Qed.

(* A step keeps the fourth part of the invariant. *)
Theorem region_step_keeps_bound (rs rs' : region_state) :
  locations_all_bound rs = true ->
  region_step rs = Region_step rs' ->
  locations_all_bound rs' = true.
Proof.
  intros Hb Hstep. apply locations_all_bound_iff in Hb.
  apply locations_all_bound_iff. revert rs' Hstep.
  apply
    (region_step_cases rs
       (fun result =>
          ∀ rs' : region_state, result = Region_step rs' -> bound rs')).
  - intros fault rs'. discriminate.
  - intros x kind d mark_kind body _ rs'.
    exact (begin_region_bound (count_uses rs) rs' x kind mark_kind body Hb).
  - intros x kind d ended_by region body _ rs'.
    exact (end_region_bound (count_uses rs) rs' x kind ended_by region body Hb).
  - intros x kind d body y region body' _ _ _ Hnot rs'.
    exact
      (local_allocation_bound (count_uses rs) rs' x y kind d region body' Hnot
         Hb).
  - intros x kind d body y region body' _ _ _ _ rs'.
    exact
      (local_cell_allocation_bound (count_uses rs) rs' x y d region body' Hb).
  - intros x kind d body _ _ _ _ rs'.
    exact (plain_step_bound (count_uses rs) rs' Hb).
  - intros bindings body He _ _ rs'.
    exact (static_step_bound (count_uses rs) rs' bindings body He Hb).
  - intros ac He rs'.
    exact (region_apply_cont_bound (count_uses rs) rs' ac He Hb).
  - intros _ _ rs'. exact (plain_step_bound (count_uses rs) rs' Hb).
Qed.

(* ======================================================================== *)
(* What a [Let] does to the heap                                            *)
(* ======================================================================== *)

Lemma eval_simples_cons_inv (vars : env) (args : list simple) (v : value)
    (vs : list value) :
  eval_simples vars args = Some (v :: vs) ->
  ∃ (a : simple) (rest : list simple),
    args = a :: rest
    ∧ eval_simple vars a = Some v
    ∧ eval_simples vars rest = Some vs.
Proof.
  destruct args as [ | a rest]; simpl; [ discriminate | ].
  destruct (eval_simple vars a) as [v0 | ] eqn:Ha; [ | discriminate ]. simpl.
  destruct (eval_simples vars rest) as [vs0 | ] eqn:Hrest; [ | discriminate ].
  simpl. intros [= <- <-]. exists a, rest. auto.
Qed.

Lemma eval_simples_nil_inv (vars : env) (args : list simple) :
  eval_simples vars args = Some [] -> args = [].
Proof.
  destruct args as [ | a rest]; simpl; [ reflexivity | ].
  destruct (eval_simple vars a); [ | discriminate ]. simpl.
  destruct (eval_simples vars rest); discriminate.
Qed.

Lemma app_last_one {A : Type} (rest : list A) (a b : A) :
  [b] = rest ++ [a] -> rest = [] ∧ a = b.
Proof.
  destruct rest as [ | r rest]; simpl.
  - intros [= ->]. auto.
  - intros [= _ H]. destruct rest; discriminate.
Qed.

Lemma app_last_two {A : Type} (rest : list A) (a b c : A) :
  [b; c] = rest ++ [a] -> rest = [b] ∧ a = c.
Proof.
  destruct rest as [ | r rest]; simpl; [ discriminate | ].
  intros [= <- H]. destruct (app_last_one rest a c H) as [-> ->]. auto.
Qed.

(* The value of a primitive of the heap, and the number of its
   arguments, by the primitive. *)
Lemma exec_prim_heap_form (statics : static_env) (hp hp' : heap)
    (q : heap_primitive) (vs : list value) (v : value) :
  exec_prim statics hp (Heap q) vs = Some (v, hp') ->
  match q with
  | Make_block_with_identity _ _ | Make_mutable_array _ =>
    v = Value_location (fresh_location hp)
  | Block_set _ _ _ =>
    v = unit_value ∧ ∃ target new : value, vs = [target; new]
  | Array_set _ _ =>
    v = unit_value ∧ ∃ target index new : value, vs = [target; index; new]
  end.
Proof.
  unfold exec_prim.
  destruct (forallb (argument_bound statics hp) vs); [ | discriminate ].
  unfold exec_bound_prim, exec_heap_primitive.
  destruct q as [mut tag | kind init field | kind | kind set_kind].
  - destruct (forallb (fun v0 => kind_eqb (kind_of_value v0) Value) vs);
      [ | discriminate ].
    intros [= <- _]. reflexivity.
  - destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
    destruct v1; try discriminate.
    destruct (lookup l hp) as [c | ]; [ | discriminate ].
    destruct c as [mut tag fields | akind elements]; try discriminate.
    destruct mut; try discriminate.
    destruct (store_field kind field v2 tag fields); [ | discriminate ].
    simpl. intros [= <- _]. split; [ reflexivity | ].
    eexists _, _. reflexivity.
  - destruct vs as [ | v1 vs]; [ discriminate | ].
    destruct (array_made kind (v1 :: vs)); [ | discriminate ].
    simpl. intros [= <- _]. reflexivity.
  - destruct vs as [ | v1 [ | v2 [ | v3 [ | v4 vs]]]]; try discriminate.
    destruct v1; try discriminate.
    destruct (lookup l hp) as [c | ]; [ | discriminate ].
    destruct c as [mut tag fields | akind elements]; try discriminate.
    destruct (array_stored set_kind akind elements v2 v3); [ | discriminate ].
    simpl. intros [= <- _]. split; [ reflexivity | ].
    eexists _, _, _. reflexivity.
Qed.

(* What a [Let] of a primitive of the heap does: it makes a cell that
   holds the values of its arguments, bound to a location that the
   heap did not bind, which is its value; or it stores the value of
   the simple that [stored] says in the cell of the location that the
   other names, and its value is unit. FlambdaHeapSpec's
   exec_prim_adds_statement, with which of the two it is. *)
Lemma exec_named_heap_prim (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (q : heap_primitive)
    (args : list simple) (v : value) :
  exec_named statics hp rd vars (Prim (Heap q) args) = Some (v, hp', rd') ->
  (is_cell_allocation (Prim (Heap q) args) = true
   ∧ ∃ (vs : list value) (c : cell),
       eval_simples vars args = Some vs
       ∧ values_of_cell c = vs
       ∧ v = Value_location (fresh_location hp)
       ∧ hp' = (fresh_location hp, c) :: hp)
  ∨ (∃ (target new_simple : simple) (l : location) (new : value)
       (c c' : cell) (i : nat),
       stored (Prim (Heap q) args) = Some (target, new_simple)
       ∧ eval_simple vars target = Some (Value_location l)
       ∧ eval_simple vars new_simple = Some new
       ∧ lookup l hp = Some c
       ∧ cell_with i new c = Some c'
       ∧ v = unit_value
       ∧ hp' = (l, c') :: hp).
Proof.
  unfold exec_named.
  destruct (eval_simples vars args) as [vs | ] eqn:Hargs; [ | discriminate ].
  simpl.
  destruct (exec_prim statics hp (Heap q) vs) as [[v0 hp0] | ] eqn:Hexec;
    [ | discriminate ].
  simpl. intros [= <- <- <-].
  pose proof (exec_prim_heap_form _ _ _ _ _ _ Hexec) as Hform.
  destruct (exec_prim_adds _ _ _ _ _ _ Hexec)
    as [(c & Hc & Hv & Hhp)
       | (l & rest & new & c & c' & i & Hvs & Hlookup & Hwith & Hv & Hhp)].
  - left. split; [ | exists vs, c; auto ].
    destruct q as [mut tag | kind init field | kind | kind set_kind];
      try reflexivity;
      destruct Hform as [Hunit _]; rewrite Hunit in Hv; discriminate.
  - right.
    destruct q as [mut tag | kind init field | kind | kind set_kind];
      try (rewrite Hv in Hform; discriminate).
    + destruct Hform as (_ & cellv & newv & Hshape). rewrite Hshape in Hvs.
      injection Hvs as -> Hrest.
      destruct (app_last_one rest new newv Hrest) as [-> ->].
      rewrite Hshape in Hargs.
      destruct (eval_simples_cons_inv _ _ _ _ Hargs)
        as (a & args1 & -> & Ha & Hargs1).
      destruct (eval_simples_cons_inv _ _ _ _ Hargs1)
        as (b & args2 & -> & Hb & Hargs2).
      rewrite (eval_simples_nil_inv _ _ Hargs2).
      exists a, b, l, newv, c, c', i. repeat split; assumption.
    + destruct Hform as (_ & cellv & index & newv & Hshape).
      rewrite Hshape in Hvs. injection Hvs as -> Hrest.
      destruct (app_last_two rest new index newv Hrest) as [-> ->].
      rewrite Hshape in Hargs.
      destruct (eval_simples_cons_inv _ _ _ _ Hargs)
        as (a & args1 & -> & Ha & Hargs1).
      destruct (eval_simples_cons_inv _ _ _ _ Hargs1)
        as (b & args2 & -> & Hb & Hargs2).
      destruct (eval_simples_cons_inv _ _ _ _ Hargs2)
        as (c0 & args3 & -> & Hc0 & Hargs3).
      rewrite (eval_simples_nil_inv _ _ Hargs3).
      exists a, c0, l, newv, c, c', i. repeat split; assumption.
Qed.

(* What a [Let] does to the heap: nothing; or it makes a cell out of
   the values of its simples; or it stores. *)
Lemma exec_named_heap_cases (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  (named_changes_heap d = false ∧ hp' = hp)
  ∨ (is_cell_allocation d = true
     ∧ ∃ (vs : list value) (c : cell),
         (∀ w : value,
           In w vs ->
           ∃ sim : simple,
             In sim (simples_of_named d) ∧ eval_simple vars sim = Some w)
         ∧ values_of_cell c = vs
         ∧ v = Value_location (fresh_location hp)
         ∧ hp' = (fresh_location hp, c) :: hp)
  ∨ (∃ (target new_simple : simple) (l : location) (new : value)
       (c c' : cell) (i : nat),
       stored d = Some (target, new_simple)
       ∧ eval_simple vars target = Some (Value_location l)
       ∧ eval_simple vars new_simple = Some new
       ∧ lookup l hp = Some c
       ∧ cell_with i new c = Some c'
       ∧ v = unit_value
       ∧ hp' = (l, c') :: hp).
Proof.
  intros Hexec. destruct (named_changes_heap d) eqn:Hchanges.
  - right.
    destruct d as [s | [u | b | va | q | rp] args | | | | | ];
      try discriminate.
    destruct (exec_named_heap_prim _ _ _ _ _ _ _ _ _ Hexec)
      as [(His & vs & c & Hargs & Hc & Hv & Hhp) | Hstore].
    + left. split; [ exact His | ]. exists vs, c.
      split; [ | auto ]. intros w Hw.
      exact (eval_simples_in vars args vs w Hargs Hw).
    + right. exact Hstore.
  - left. split; [ reflexivity | ].
    apply (exec_named_heap statics hp hp' rd rd' vars d v); [ | exact Hexec ].
    intros q args ->. discriminate.
Qed.

(* ======================================================================== *)
(* The cells that a state lists                                             *)
(* ======================================================================== *)

(* A location that may be evaluated is of the heap, or the cell is in
   a frame. *)
Lemma usable_location_near (rs : region_state) (l : location) :
  value_usable rs (Value_location l) = true ->
  heap_location rs l || cell_in rs (rs_frames rs) l = true.
Proof.
  unfold value_usable, cell_dead, heap_location, cell_in.
  destruct (lookup l (rs_cells rs)) as [m | ]; simpl; [ | reflexivity ].
  destruct (on_stack m (rs_frames rs)); simpl; congruence.
Qed.

(* One that is in no frame is of the heap. *)
Lemma usable_not_local_heap_location (rs : region_state) (l : location) :
  value_usable rs (Value_location l) = true ->
  value_local rs (Value_location l) = false ->
  heap_location rs l = true.
Proof.
  unfold value_usable, value_local, cell_dead, local_cell, heap_location.
  destruct (lookup l (rs_cells rs)) as [m | ]; simpl; [ | reflexivity ].
  destruct (on_stack m (rs_frames rs)); simpl; congruence.
Qed.

(* A location of the heap may be evaluated, and so may one of a cell
   in a frame. *)
Lemma heap_location_usable (rs : region_state) (l : location) :
  heap_location rs l = true -> value_usable rs (Value_location l) = true.
Proof.
  unfold value_usable, cell_dead, heap_location.
  destruct (lookup l (rs_cells rs)) as [m | ]; simpl;
    [ discriminate | reflexivity ].
Qed.

Lemma cell_in_usable (rs : region_state) (above frames : list frame)
    (l : location) :
  rs_frames rs = above ++ frames ->
  cell_in rs frames l = true ->
  value_usable rs (Value_location l) = true.
Proof.
  unfold value_usable, cell_dead, cell_in. intros ->.
  destruct (lookup l (rs_cells rs)) as [m | ]; [ | discriminate ].
  intros H. rewrite on_stack_app, H, orb_true_r. reflexivity.
Qed.

(* The value of a simple that may be evaluated and is in no frame holds
   locations of the heap only. *)
Lemma simple_holds_heap_cells (rs : region_state) (sim : simple) (v : value) :
  forallb (fun '(_, v) => value_kept_cells (heap_location rs) v)
    (state_vars (rs_state rs))
  = true ->
  simple_usable rs sim = true ->
  simple_local rs sim = false ->
  eval_simple (state_vars (rs_state rs)) sim = Some v ->
  value_holds_cells (heap_location rs) v = true.
Proof.
  intros Hvars Hu Hl Hv. unfold simple_usable, simple_local in Hu, Hl.
  rewrite Hv in Hu, Hl.
  apply value_kept_cells_holds;
    [ | exact (eval_simple_kept_cells _ _ sim v Hvars Hv) ].
  intros l ->. exact (usable_not_local_heap_location rs l Hu Hl).
Qed.

(* The value of a simple that may be evaluated is what a frame, or a
   cell in the newest frame, may hold of locations. *)
Lemma simple_placed_cells (rs : region_state) (sim : simple) (v : value) :
  forallb (fun '(_, v) => value_kept_cells (heap_location rs) v)
    (state_vars (rs_state rs))
  = true ->
  simple_usable rs sim = true ->
  eval_simple (state_vars (rs_state rs)) sim = Some v ->
  value_placed_cells
    (fun l => heap_location rs l || cell_in rs (rs_frames rs) l)
    (heap_location rs) v
  = true.
Proof.
  intros Hvars Hu Hv. unfold simple_usable in Hu. rewrite Hv in Hu.
  apply value_kept_cells_placed;
    [ | exact (eval_simple_kept_cells _ _ sim v Hvars Hv) ].
  intros l ->. exact (usable_location_near rs l Hu).
Qed.

(* The cells that two states list are the same at the locations that a
   heap binds. *)
Definition cells_same_on (hp : heap) (rs rs' : region_state) : Prop :=
  ∀ l : location,
    is_bound l hp = true -> lookup l (rs_cells rs') = lookup l (rs_cells rs).

Lemma cells_same_on_same (hp : heap) (rs rs' : region_state) :
  rs_cells rs' = rs_cells rs -> cells_same_on hp rs rs'.
Proof. intros Hcells l _. rewrite Hcells. reflexivity. Qed.

Lemma value_holds_cells_same (hp : heap) (rs rs' : region_state) (v : value) :
  cells_same_on hp rs rs' ->
  value_closed hp v ->
  value_holds_cells (heap_location rs) v = true ->
  value_holds_cells (heap_location rs') v = true.
Proof.
  intros Hsame Hclosed. apply value_holds_cells_in. intros l Hl.
  unfold heap_location. rewrite (Hsame l (Hclosed l Hl)). auto.
Qed.

Lemma value_kept_cells_same (hp : heap) (rs rs' : region_state) (v : value) :
  cells_same_on hp rs rs' ->
  value_closed hp v ->
  value_kept_cells (heap_location rs) v = true ->
  value_kept_cells (heap_location rs') v = true.
Proof.
  intros Hsame Hclosed. apply value_kept_cells_in. intros l Hl.
  unfold heap_location. rewrite (Hsame l (Hclosed l Hl)). auto.
Qed.

Lemma value_placed_cells_same (hp : heap) (rs rs' : region_state)
    (frames frames' : list frame) (v : value) :
  cells_same_on hp rs rs' ->
  map frame_mark frames' = map frame_mark frames ->
  value_closed hp v ->
  value_placed_cells (fun l => heap_location rs l || cell_in rs frames l)
    (heap_location rs) v
  = true ->
  value_placed_cells (fun l => heap_location rs' l || cell_in rs' frames' l)
    (heap_location rs') v
  = true.
Proof.
  intros Hsame Hmarks Hclosed.
  apply value_placed_cells_in; intros l Hl; unfold heap_location, cell_in;
    rewrite (Hsame l (Hclosed l Hl)); [ | auto ].
  destruct (lookup l (rs_cells rs)) as [m | ]; simpl; [ | auto ].
  rewrite (on_stack_same_marks m frames' frames Hmarks). auto.
Qed.

(* ======================================================================== *)
(* What the frames hold, of locations                                       *)
(* ======================================================================== *)

Lemma frames_hold_cells_app (rs : region_state) (above rest : list frame) :
  frames_hold_cells rs (above ++ rest) = true ->
  frames_hold_cells rs rest = true.
Proof.
  induction above as [ | f above IH]; simpl; [ auto | ].
  intros H. apply andb_true_iff in H as [_ H]. exact (IH H).
Qed.

Lemma frames_hold_cells_same (hp : heap) (rs rs' : region_state)
    (frames : list frame) :
  cells_same_on hp rs rs' ->
  (∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
    In (s, allocated) (live_locals frames) ->
    In w (allocated_values allocated) ->
    value_closed hp w) ->
  frames_hold_cells rs frames = true ->
  frames_hold_cells rs' frames = true.
Proof.
  intros Hsame. induction frames as [ | f below IH]; simpl; [ reflexivity | ].
  intros Hclosed H. apply andb_true_iff in H as [Hf Hbelow].
  apply andb_true_iff. split.
  - apply forallb_forall. intros [s allocated] Hin.
    apply forallb_forall. intros w Hw.
    pose proof (forallb_in _ _ (s, allocated) Hf Hin) as Hall.
    cbv beta iota in Hall. pose proof (forallb_in _ _ w Hall Hw) as Hplaced.
    apply
      (value_placed_cells_same hp rs rs' (f :: below) (f :: below) w Hsame
         eq_refl);
      [ | exact Hplaced ].
    apply (Hclosed s allocated w);
      [ apply in_or_app; left; exact Hin
      | rewrite <- values_of_static_values; exact Hw ].
  - apply IH; [ | exact Hbelow ]. intros s allocated w Hin.
    apply (Hclosed s allocated w), in_or_app. right. exact Hin.
Qed.

(* What a frame binds a symbol to holds directly what a variable may
   hold, of locations. *)
Lemma frames_hold_cells_in (rs : region_state) (frames : list frame)
    (s : symbol) (allocated : statically_allocated_value) (w : value) :
  frames_hold_cells rs frames = true ->
  In (s, allocated) (live_locals frames) ->
  In w (allocated_values allocated) ->
  value_kept_cells (heap_location rs) w = true.
Proof.
  induction frames as [ | f below IH]; simpl; [ intros _ [] | ].
  intros H Hin Hw. apply andb_true_iff in H as [Hf Hbelow].
  apply in_app_iff in Hin as [Hin | Hin]; [ | exact (IH Hbelow Hin Hw) ].
  pose proof (forallb_in _ _ (s, allocated) Hf Hin) as Hall.
  cbv beta iota in Hall. rewrite values_of_static_values in Hall.
  exact (value_placed_cells_kept _ _ w (forallb_in _ _ w Hall Hw)).
Qed.

(* ======================================================================== *)
(* The third part of the invariant, in parts                                *)
(* ======================================================================== *)

(* The test of the variables and of the continuations, as
   FlambdaRegionsStepsProofs.vals_ok. *)
Definition kept_cells (rs : region_state) (v : value) : Prop :=
  value_kept_cells (heap_location rs) v = true.

Lemma kept_cells_vals (rs : region_state) (s : state) :
  forallb (fun '(_, v) => value_kept_cells (heap_location rs) v)
    (state_vars s)
  = true
  ∧ forallb (fun '(_, c) => cont_kept_cells (heap_location rs) c)
      (state_conts s)
    = true
  <-> vals_ok (kept_cells rs) s.
Proof.
  unfold vals_ok, kept_cells, env_values_all.
  destruct
    (forallb_pairs_Forall (value_kept_cells (heap_location rs)) (state_vars s))
    as [Hv1 Hv2].
  destruct (conts_kept_cells_split (heap_location rs) (state_conts s))
    as [Hc1 Hc2].
  split; intros [Hv Hc]; split; auto.
Qed.

Lemma kept_cells_partial (rs : region_state) (callee : value)
    (args : list value) (args_arity : arity) :
  kept_cells rs (Value_partial_application callee args args_arity)
  <-> kept_cells rs callee ∧ Forall (kept_cells rs) args.
Proof.
  unfold kept_cells. simpl. split.
  - intros H. apply andb_true_iff in H as [Hcallee Hargs].
    split; [ exact Hcallee | ].
    exact (proj1 (forallb_Forall _ args) Hargs).
  - intros [Hcallee Hargs]. apply andb_true_iff.
    split; [ exact Hcallee | ].
    exact
      (proj2 (forallb_Forall (value_kept_cells (heap_location rs)) args)
         Hargs).
Qed.

(* [cells_live], part by part. The fourth is
   FlambdaRegionsFreshProofs.cells_below. *)
Lemma cells_live_parts (rs : region_state) :
  cells_live rs = true
  <-> distinct (map frame_mark (rs_frames rs)) = true
      ∧ (∀ (l : location) (m : mark),
          In (l, m) (rs_cells rs) ->
          is_bound l (state_heap (rs_state rs)) = true
          ∧ (rs_base rs <= m)%positive
          ∧ (m < rs_next rs)%positive)
      ∧ (∀ (l : location) (c : cell),
          In (l, c) (state_heap (rs_state rs)) -> cell_holds rs l c = true)
      ∧ cells_below (rs_next rs) (state_heap (rs_state rs))
      ∧ (∀ (s : symbol) (allocated : statically_allocated_value)
           (w : value),
          In (s, allocated) (static_values (state_statics (rs_state rs))) ->
          In w (allocated_values allocated) ->
          value_holds_cells (heap_location rs) w = true)
      ∧ frames_hold_cells rs (rs_frames rs) = true
      ∧ vals_ok (kept_cells rs) (rs_state rs).
Proof.
  unfold cells_live. cbv zeta. split.
  - intros H.
    apply andb_true_iff in H as [H L8]. apply andb_true_iff in H as [H L7].
    apply andb_true_iff in H as [H L6]. apply andb_true_iff in H as [H L5].
    apply andb_true_iff in H as [H L4]. apply andb_true_iff in H as [H L3].
    apply andb_true_iff in H as [L1 L2].
    split; [ exact L1 | ]. split.
    { intros l m Hin. pose proof (forallb_in _ _ (l, m) L2 Hin) as H.
      cbv beta iota in H. apply andb_true_iff in H as [Hb Hm].
      apply symbol_between_iff in Hm. exact (conj Hb Hm). }
    split; [ intros l c Hin; exact (forallb_in _ _ (l, c) L3 Hin) | ].
    split; [ exact (proj1 (cells_below_forallb _ _) L4) | ]. split.
    { intros s allocated w Hin Hw.
      pose proof (forallb_in _ _ (s, allocated) L5 Hin) as H.
      cbv beta iota in H. rewrite values_of_static_values in H.
      exact (forallb_in _ _ w H Hw). }
    split; [ exact L6 | ].
    exact (proj1 (kept_cells_vals rs (rs_state rs)) (conj L7 L8)).
  - intros (L1 & L2 & L3 & L4 & L5 & L6 & L7).
    apply kept_cells_vals in L7 as [L7 L8].
    repeat (apply andb_true_iff; split).
    + exact L1.
    + apply forallb_forall. intros [l m] Hin.
      destruct (L2 l m Hin) as [Hb Hm]. apply andb_true_iff.
      split; [ exact Hb | apply symbol_between_iff; exact Hm ].
    + apply forallb_forall. intros [l c] Hin. exact (L3 l c Hin).
    + exact (proj2 (cells_below_forallb _ _) L4).
    + apply forallb_forall. intros [s allocated] Hin.
      apply forallb_forall. intros w Hw.
      rewrite values_of_static_values in Hw. exact (L5 s allocated w Hin Hw).
    + exact L6.
    + exact L7.
    + exact L8.
Qed.
