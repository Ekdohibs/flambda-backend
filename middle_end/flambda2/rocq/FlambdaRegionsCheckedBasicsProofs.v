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

(* Lemmas for the proofs of the statements of FlambdaRegionsCheckedSpec.v:
   first facts about its definitions. A value within a bound may be
   evaluated; the open regions of a state that runs are bound to marks that
   are on the stack; what the frames and the static environment bind
   reaches only symbols that may be evaluated. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
Import ListNotations.

Lemma Forall2_in_right {A B : Type} (R : A -> B -> Prop) (l : list A)
    (l' : list B) (b : B) :
  Forall2 R l l' -> In b l' -> ∃ a : A, In a l ∧ R a b.
Proof.
  induction 1 as [ | a0 b0 l l' Hab _ IH]; intros Hin; [ contradiction | ].
  destruct Hin as [<- | Hin].
  - exists a0. split; [ left; reflexivity | exact Hab ].
  - destruct (IH Hin) as (a & Ha & Hr).
    exists a. split; [ right; exact Ha | exact Hr ].
Qed.

Lemma value_all_mono (ok ok' : symbol -> bool) (v : value) :
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  value_all ok v = true ->
  value_all ok' v = true.
Proof.
  intros Hok.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    simpl; auto.
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) H Hin)).
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) H Hin)).
  - intros H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH. exact (IH w Hin (forallb_in _ _ w H Hin)).
  - intros H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH. exact (IH w Hin (forallb_in _ _ w H Hin)).
  - intros H. apply andb_true_iff in H as [Hcallee Hargs].
    apply andb_true_iff. split; [ exact (IHcallee Hcallee) | ].
    apply forallb_forall. intros arg Hin.
    rewrite Forall_forall in IHargs.
    exact (IHargs _ Hin (forallb_in _ _ arg Hargs Hin)).
Qed.

(* A symbol that a frame binds may be evaluated. *)
Lemma local_live (rs : region_state) (s : symbol) :
  local_in (rs_frames rs) s = true -> live_symbol rs s = true.
Proof.
  unfold local_in, live_symbol, value_usable, live_statics.
  cbn [static_values]. rewrite is_some_lookup_app. intros ->.
  apply orb_true_r.
Qed.

(* A symbol within a bound may be evaluated. *)
Lemma within_live (rs : region_state) (stop : option symbol) (s : symbol) :
  symbol_within rs stop s = true -> live_symbol rs s = true.
Proof.
  unfold symbol_within. intros H. apply orb_true_iff in H as [H | H].
  - exact (heap_live rs s H).
  - apply andb_true_iff in H as [H _]. exact (local_live rs s H).
Qed.

Lemma value_within_live (rs : region_state) (stop : option symbol)
    (v : value) :
  value_within rs stop v = true -> value_all (live_symbol rs) v = true.
Proof. apply value_all_mono. exact (within_live rs stop). Qed.

(* A value that reaches symbols of the heap only is within every
   bound. *)
Lemma heap_within (rs : region_state) (stop : option symbol) (v : value) :
  value_all (heap_symbol rs) v = true -> value_within rs stop v = true.
Proof.
  apply value_all_mono. intros s H. unfold symbol_within. rewrite H.
  reflexivity.
Qed.

(* A value all of whose symbols may be evaluated may be evaluated
   (decision D45). *)
Lemma live_usable (rs : region_state) (v : value) :
  value_all (live_symbol rs) v = true -> value_usable rs v = true.
Proof. destruct v; try reflexivity. cbn [value_all]. exact (fun H => H). Qed.

(* A value that reaches symbols of the heap only is not local (decision
   D46). *)
Lemma heap_value_not_local (rs : region_state) (v : value) :
  symbols_fresh rs = true ->
  value_all (heap_symbol rs) v = true ->
  value_local rs v = false.
Proof.
  intros Hfresh. destruct v; try reflexivity. cbn [value_all]. intros Hheap.
  pose proof (heap_not_local rs s Hfresh Hheap) as H.
  unfold not_local_symbol in H. apply andb_true_iff in H as [_ H].
  destruct (value_local rs (Value_symbol s)); [ discriminate | reflexivity ].
Qed.

(* The marks of the open regions: each open region is bound to its
   mark. *)
Lemma open_marks_in (vars : env) (opens : list variable) :
  ∀ (ms : list mark) (r : variable),
    open_marks vars opens = Some ms ->
    among r opens = true ->
    ∃ m : mark, lookup r vars = Some (Value_symbol m) ∧ In m ms.
Proof.
  induction opens as [ | x opens IH]; intros ms r; cbn [open_marks among
    existsb]; [ discriminate | ].
  destruct (lookup x vars) as [[ | | | m | | | | | ] | ] eqn:Hx;
    try discriminate.
  destruct (open_marks vars opens) as [ms' | ] eqn:Hms; [ | discriminate ].
  cbn. intros [= <-].
  destruct (Pos.eqb r x) eqn:Heq; cbn [orb].
  - intros _. apply Pos.eqb_eq in Heq. subst x. exists m.
    split; [ exact Hx | left; reflexivity ].
  - intros Hmem. destruct (IH ms' r eq_refl Hmem) as (m' & Hr & Hin).
    exists m'. split; [ exact Hr | right; exact Hin ].
Qed.

(* An open region has a place, and its mark is the one that its
   variable is bound to. *)
Lemma region_place_open (vars : env) (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable),
    open_marks vars opens = Some ms ->
    among r opens = true ->
    ∃ (m : mark) (stop : option symbol),
      region_place opens ms above r = Some (m, stop) ∧ In m ms.
Proof.
  induction opens as [ | x opens IH]; intros ms above r; cbn [open_marks among
    existsb region_place]; [ discriminate | ].
  destruct (lookup x vars) as [[ | | | m | | | | | ] | ]; try discriminate.
  destruct (open_marks vars opens) as [ms' | ] eqn:Hms; [ | discriminate ].
  cbn. intros [= <-]. rewrite (Pos.eqb_sym x r).
  destruct (Pos.eqb r x); cbn [orb].
  - intros _. exists m, above. split; [ reflexivity | left; reflexivity ].
  - intros Hmem.
    destruct (IH ms' (Some m) r eq_refl Hmem) as (m' & stop & Hplace & Hin).
    exists m', stop. split; [ exact Hplace | right; exact Hin ].
Qed.

Lemma marks_placed_in (frames : list frame) (ms : list mark) (m : mark) :
  marks_placed frames ms = true -> In m ms -> on_stack m frames = true.
Proof.
  unfold marks_placed. intros H Hin. apply andb_true_iff in H as [H _].
  exact (forallb_in _ _ m H Hin).
Qed.

(* The simples that a step evaluates name no symbol that the machine
   drew. *)
Lemma head_simples_below (rs : region_state) (s : simple) :
  symbols_fresh rs = true ->
  In s (evaluated_simples (state_expr (rs_state rs))) ->
  simple_below (rs_base rs) s = true.
Proof.
  intros Hfresh Hin.
  exact (evaluated_simples_below _ _ s (symbols_fresh_expr rs Hfresh) Hin).
Qed.

(* Decision D45, of the state that runs: a simple of the head that the
   check lets be evaluated has a value within no bound: all of whose
   symbols may be evaluated. *)
Lemma running_usable_within (rs : region_state) (cs : check_state)
    (ms : list mark) (s : simple) (v : value) :
  running_checked rs cs ms ->
  simple_below (rs_base rs) s = true ->
  check_usable cs s = true ->
  eval_simple (state_vars (rs_state rs)) s = Some v ->
  value_within rs None v = true.
Proof.
  intros (Hms & Hplaced & _ & Henv) Hin. unfold check_usable.
  assert (Hweaken : ∀ (stop : option symbol) (w : value),
            value_within rs stop w = true -> value_within rs None w = true).
  { intros stop w. apply value_all_mono. intros s0. unfold symbol_within.
    destruct (heap_symbol rs s0); [ reflexivity | ]. cbn [orb].
    intros H. apply andb_true_iff in H as [-> _]. reflexivity. }
  destruct s as [x | sym | c]; cbn [lifetime_of eval_simple].
  - destruct (lookup x (cs_env cs)) as [l | ] eqn:Hl; [ | discriminate ].
    intros Husable Hv. specialize (Henv x l v Hl Hv).
    destruct l as [ | | r | ]; cbn [lifetime_usable lifetime_holds] in *.
    + exact (heap_within rs None v Henv).
    + exact (Hweaken _ v Henv).
    + destruct (region_place_open _ _ ms None r Hms Husable)
        as (m & stop & Hplace & Hinm).
      rewrite Hplace in Henv.
      exact (Hweaken _ v (Henv (marks_placed_in _ _ _ Hplaced Hinm))).
    + discriminate.
  - intros _ [= <-]. unfold value_within. cbn [value_all].
    unfold symbol_within. cbn [simple_below] in Hin.
    rewrite (below_base_heap rs sym Hin). reflexivity.
  - intros _ [= <-]. reflexivity.
Qed.

(* Decision D46, of the state that runs: a simple that the check takes
   to live for ever has a value that reaches symbols of the heap
   only. *)
Lemma running_forever_heap (rs : region_state) (cs : check_state)
    (ms : list mark) (s : simple) (v : value) :
  running_checked rs cs ms ->
  simple_below (rs_base rs) s = true ->
  is_forever (lifetime_of cs s) = true ->
  eval_simple (state_vars (rs_state rs)) s = Some v ->
  value_all (heap_symbol rs) v = true.
Proof.
  intros (_ & _ & _ & Henv) Hin.
  destruct s as [x | sym | c]; cbn [lifetime_of eval_simple].
  - destruct (lookup x (cs_env cs)) as [l | ] eqn:Hl; [ | discriminate ].
    intros Hforever Hv. specialize (Henv x l v Hl Hv).
    destruct l; cbn [is_forever lifetime_holds] in *; try discriminate.
    exact Henv.
  - intros _ [= <-]. exact (below_base_heap rs sym Hin).
  - intros _ [= <-]. reflexivity.
Qed.

(* The mark of a frame may not be evaluated, and is on the stack. *)
Lemma on_stack_frame (m : mark) (frames : list frame) :
  on_stack m frames = true -> ∃ f : frame, In f frames ∧ frame_mark f = m.
Proof.
  unfold on_stack. intros H. apply existsb_exists in H as (f & Hin & Heq).
  apply Pos.eqb_eq in Heq. exists f. split; assumption.
Qed.

Lemma mark_open (rs : region_state) (m : mark) :
  symbols_fresh rs = true ->
  on_stack m (rs_frames rs) = true ->
  value_open_mark rs (Value_symbol m) = true.
Proof.
  intros Hfresh Hon. unfold value_open_mark. rewrite Hon, andb_true_r.
  destruct (on_stack_frame m _ Hon) as (f & Hin & <-).
  apply symbols_fresh_parts in Hfresh
    as (_ & _ & _ & _ & Hdrawn & Hunbound & _).
  destruct (Hdrawn f Hin) as (Hbase & _ & _).
  unfold value_usable, drawn, live_statics. cbn [static_values].
  rewrite (Hunbound f Hin).
  destruct (Pos.leb_spec (rs_base rs) (frame_mark f)); [ reflexivity | lia ].
Qed.

(* Decision D47, of the state that runs: the variable of a region that
   the check takes to be open has for its value a mark that is on the
   stack. *)
Lemma running_region_open (rs : region_state) (cs : check_state)
    (ms : list mark) (r : variable) :
  running_checked rs cs ms ->
  among r (cs_open cs) = true ->
  ∃ m : mark,
    lookup r (state_vars (rs_state rs)) = Some (Value_symbol m)
    ∧ on_stack m (rs_frames rs) = true.
Proof.
  intros (Hms & Hplaced & _) Hmem.
  destruct (open_marks_in _ _ ms r Hms Hmem) as (m & Hr & Hin).
  exists m. split; [ exact Hr | exact (marks_placed_in _ _ _ Hplaced Hin) ].
Qed.

(* What the frames and the static environment bind reaches only
   symbols that may be evaluated. *)
Lemma allocated_all_mono (ok ok' : symbol -> bool)
    (allocated : statically_allocated_value) :
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  allocated_all ok allocated = true ->
  allocated_all ok' allocated = true.
Proof.
  intros Hok.
  destruct allocated
    as [code slots | slot0 symbols decls slots | tag fields | contents
       | bytes | elements | c];
    cbn [allocated_all]; try reflexivity.
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    exact (value_all_mono _ _ w Hok (forallb_in _ _ (vslot, w) H Hin)).
  - intros H. apply andb_true_iff in H as [Hsymbols Hslots].
    apply andb_true_iff. split.
    + apply forallb_forall. intros [fslot s] Hin.
      exact (Hok s (forallb_in _ _ (fslot, s) Hsymbols Hin)).
    + apply forallb_forall. intros [vslot w] Hin.
      exact (value_all_mono _ _ w Hok (forallb_in _ _ (vslot, w) Hslots Hin)).
  - intros H. apply forallb_forall. intros w Hin.
    exact (value_all_mono _ _ w Hok (forallb_in _ _ w H Hin)).
  - exact (value_all_mono _ _ _ Hok).
  - intros H. apply forallb_forall. intros w Hin.
    exact (value_all_mono _ _ w Hok (forallb_in _ _ w H Hin)).
Qed.

Lemma local_in_below (above rest : list frame) (s : symbol) :
  local_in rest s = true -> local_in (above ++ rest) s = true.
Proof.
  unfold local_in. rewrite live_locals_app, is_some_lookup_app. intros ->.
  apply orb_true_r.
Qed.

Lemma frames_deep_live (rs : region_state) :
  ∀ above rest : list frame,
    rs_frames rs = above ++ rest ->
    frames_deep rs rest = true ->
    forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
      (live_locals rest)
    = true.
Proof.
  intros above rest. revert above.
  induction rest as [ | f below IH]; intros above Hframes;
    [ reflexivity | ].
  cbn [frames_deep]. intros H. apply andb_true_iff in H as [Hf Hbelow].
  change (live_locals (f :: below)) with (frame_locals f ++ live_locals below).
  rewrite forallb_app. apply andb_true_iff. split.
  - apply forallb_forall. intros [s allocated] Hin.
    pose proof (forallb_in _ _ (s, allocated) Hf Hin) as Hall.
    cbv beta iota in Hall. revert Hall. apply allocated_all_mono.
    intros s0 H0. apply orb_true_iff in H0 as [H0 | H0].
    + exact (heap_live rs s0 H0).
    + apply local_live. rewrite Hframes. exact (local_in_below above _ s0 H0).
  - apply (IH (above ++ [f])); [ | exact Hbelow ].
    rewrite Hframes, <- app_assoc. reflexivity.
Qed.

Lemma bindings_live (rs : region_state) :
  frames_deep rs (rs_frames rs) = true ->
  statics_deep rs = true ->
  forallb (fun '(_, allocated) => allocated_all (live_symbol rs) allocated)
    (static_values (live_statics rs))
  = true.
Proof.
  intros Hframes Hstatics. unfold live_statics. cbn [static_values].
  rewrite forallb_app. apply andb_true_iff. split.
  - exact (frames_deep_live rs [] (rs_frames rs) eq_refl Hframes).
  - unfold statics_deep in Hstatics. apply forallb_forall.
    intros [s allocated] Hin.
    pose proof (forallb_in _ _ (s, allocated) Hstatics Hin) as Hall.
    cbv beta iota in Hall. revert Hall. apply allocated_all_mono.
    exact (heap_live rs).
Qed.
