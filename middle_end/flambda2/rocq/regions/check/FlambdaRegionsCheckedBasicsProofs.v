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
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
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

Lemma value_all_cells_mono (ok ok' : location -> bool) (v : value) :
  (∀ l : location, ok l = true -> ok' l = true) ->
  value_all_cells ok v = true ->
  value_all_cells ok' v = true.
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

(* Every location that a value reaches is one of [ok]: said of the
   list of its locations. *)
Lemma value_all_cells_iff (ok : location -> bool) (v : value) :
  value_all_cells ok v = true
  <-> ∀ l : location, In l (locations_of_value v) -> ok l = true.
Proof.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    cbn [value_all_cells locations_of_value];
    try (split; [ intros _ l0 [] | reflexivity ]).
  - rewrite Forall_forall in IH. rewrite forallb_forall. split.
    + intros H l Hl. apply in_flat_map in Hl as ([vslot w] & Hin & Hl).
      exact (proj1 (IH (vslot, w) Hin) (H (vslot, w) Hin) l Hl).
    + intros H [vslot w] Hin. apply (proj2 (IH (vslot, w) Hin)).
      intros l Hl. apply H, in_flat_map. exists (vslot, w).
      split; assumption.
  - rewrite Forall_forall in IH. rewrite forallb_forall. split.
    + intros H l Hl. apply in_flat_map in Hl as ([vslot w] & Hin & Hl).
      exact (proj1 (IH (vslot, w) Hin) (H (vslot, w) Hin) l Hl).
    + intros H [vslot w] Hin. apply (proj2 (IH (vslot, w) Hin)).
      intros l Hl. apply H, in_flat_map. exists (vslot, w).
      split; assumption.
  - rewrite Forall_forall in IH. rewrite forallb_forall. split.
    + intros H l Hl. apply in_flat_map in Hl as (w & Hin & Hl).
      exact (proj1 (IH w Hin) (H w Hin) l Hl).
    + intros H w Hin. apply (proj2 (IH w Hin)).
      intros l Hl. apply H, in_flat_map. exists w. split; assumption.
  - rewrite Forall_forall in IH. rewrite forallb_forall. split.
    + intros H l Hl. apply in_flat_map in Hl as (w & Hin & Hl).
      exact (proj1 (IH w Hin) (H w Hin) l Hl).
    + intros H w Hin. apply (proj2 (IH w Hin)).
      intros l Hl. apply H, in_flat_map. exists w. split; assumption.
  - split.
    + intros H l0 [<- | []]. exact H.
    + intros H. exact (H l (or_introl eq_refl)).
  - rewrite Forall_forall in IHargs. rewrite andb_true_iff, forallb_forall.
    split.
    + intros [Hcallee Hargs] l Hl. apply in_app_or in Hl as [Hl | Hl].
      * exact (proj1 IHcallee Hcallee l Hl).
      * apply in_flat_map in Hl as (w & Hin & Hl).
        exact (proj1 (IHargs w Hin) (Hargs w Hin) l Hl).
    + intros H. split.
      * apply (proj2 IHcallee). intros l Hl. apply H, in_or_app. left.
        exact Hl.
      * intros w Hin. apply (proj2 (IHargs w Hin)). intros l Hl.
        apply H, in_or_app. right. apply in_flat_map. exists w.
        split; assumption.
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
Proof.
  unfold value_within. intros H. apply andb_true_iff in H as [H _].
  revert H. apply value_all_mono. exact (within_live rs stop).
Qed.

(* A location of the heap is within every bound. *)
Lemma heap_cell_within (rs : region_state) (stop : option symbol)
    (l : location) :
  heap_location rs l = true -> cell_within rs stop l = true.
Proof. unfold cell_within. intros ->. reflexivity. Qed.

(* A value that reaches symbols and locations of the heap only is
   within every bound. *)
Lemma heap_within (rs : region_state) (stop : option symbol) (v : value) :
  value_for_ever rs v = true -> value_within rs stop v = true.
Proof.
  unfold value_for_ever, value_within. intros H.
  apply andb_true_iff in H as [Hsymbols Hcells]. apply andb_true_iff. split.
  - revert Hsymbols. apply value_all_mono. intros s H. unfold symbol_within.
    rewrite H. reflexivity.
  - revert Hcells. apply value_all_cells_mono.
    exact (heap_cell_within rs stop).
Qed.

(* The two tests of a partial application and of a block, by the
   parts. *)
Lemma all_both_partial (ok : symbol -> bool) (ok_cell : location -> bool)
    (callee : value) (args : list value) (args_arity : arity) :
  value_all ok (Value_partial_application callee args args_arity)
  && value_all_cells ok_cell
       (Value_partial_application callee args args_arity)
  = true
  <-> value_all ok callee && value_all_cells ok_cell callee = true
      ∧ Forall (fun v => value_all ok v && value_all_cells ok_cell v = true)
          args.
Proof.
  cbn [value_all value_all_cells].
  rewrite !andb_true_iff, !forallb_forall, Forall_forall. split.
  - intros [[Hc Ha] [Hc' Ha']]. split; [ split; assumption | ].
    intros arg Hin. rewrite andb_true_iff.
    exact (conj (Ha arg Hin) (Ha' arg Hin)).
  - intros [[Hc Hc'] Hargs]. split; split; try assumption.
    + intros arg Hin. pose proof (Hargs arg Hin) as H.
      apply andb_true_iff in H as [H _]. exact H.
    + intros arg Hin. pose proof (Hargs arg Hin) as H.
      apply andb_true_iff in H as [_ H]. exact H.
Qed.

Lemma all_both_block (ok : symbol -> bool) (ok_cell : location -> bool)
    (tag : Z) (fields : list value) :
  value_all ok (Value_block tag fields)
  && value_all_cells ok_cell (Value_block tag fields)
  = true
  <-> Forall (fun v => value_all ok v && value_all_cells ok_cell v = true)
        fields.
Proof.
  cbn [value_all value_all_cells].
  rewrite !andb_true_iff, !forallb_forall, Forall_forall. split.
  - intros [H H'] w Hin. rewrite andb_true_iff.
    exact (conj (H w Hin) (H' w Hin)).
  - intros H. split; intros w Hin; pose proof (H w Hin) as Hw;
      apply andb_true_iff in Hw as [Hw Hw']; assumption.
Qed.

Lemma for_ever_partial (rs : region_state) (callee : value)
    (args : list value) (args_arity : arity) :
  value_for_ever rs (Value_partial_application callee args args_arity) = true
  <-> value_for_ever rs callee = true
      ∧ Forall (fun v => value_for_ever rs v = true) args.
Proof. exact (all_both_partial _ _ callee args args_arity). Qed.

Lemma for_ever_block (rs : region_state) (tag : Z) (fields : list value) :
  value_for_ever rs (Value_block tag fields) = true
  <-> Forall (fun v => value_for_ever rs v = true) fields.
Proof. exact (all_both_block _ _ tag fields). Qed.

Lemma within_partial (rs : region_state) (stop : option symbol)
    (callee : value) (args : list value) (args_arity : arity) :
  value_within rs stop (Value_partial_application callee args args_arity)
  = true
  <-> value_within rs stop callee = true
      ∧ Forall (fun v => value_within rs stop v = true) args.
Proof. exact (all_both_partial _ _ callee args args_arity). Qed.

Lemma within_block (rs : region_state) (stop : option symbol) (tag : Z)
    (fields : list value) :
  value_within rs stop (Value_block tag fields) = true
  <-> Forall (fun v => value_within rs stop v = true) fields.
Proof. exact (all_both_block _ _ tag fields). Qed.

(* A value that is within a bound is within no bound. *)
Lemma within_no_bound (rs : region_state) (stop : option symbol)
    (v : value) :
  value_within rs stop v = true -> value_within rs None v = true.
Proof.
  unfold value_within. intros H.
  apply andb_true_iff in H as [Hsymbols Hcells]. apply andb_true_iff. split.
  - revert Hsymbols. apply value_all_mono. intros s0. unfold symbol_within.
    destruct (heap_symbol rs s0); [ reflexivity | ]. cbn [orb].
    intros H. apply andb_true_iff in H as [-> _]. reflexivity.
  - revert Hcells. apply value_all_cells_mono. intros l. unfold cell_within.
    destruct (heap_location rs l); [ reflexivity | ]. cbn [orb].
    destruct (lookup l (rs_cells rs)) as [m | ]; [ | exact (fun H => H) ].
    intros H. apply andb_true_iff in H as [-> _]. reflexivity.
Qed.

(* A location within a bound is of a cell that is not dead. *)
Lemma cell_within_not_dead (rs : region_state) (stop : option symbol)
    (l : location) :
  cell_within rs stop l = true -> cell_dead rs l = false.
Proof.
  unfold cell_within, heap_location, cell_dead.
  destruct (lookup l (rs_cells rs)) as [m | ]; cbn [is_some negb orb];
    [ | reflexivity ].
  intros H. apply andb_true_iff in H as [-> _]. reflexivity.
Qed.

(* A value within a bound may be evaluated (decision D45). *)
Lemma within_usable (rs : region_state) (stop : option symbol) (v : value) :
  value_within rs stop v = true -> value_usable rs v = true.
Proof.
  unfold value_within. intros H. apply andb_true_iff in H as [Hsymbols Hcells].
  destruct v; try reflexivity.
  - cbn [value_all] in Hsymbols. exact (within_live rs stop s Hsymbols).
  - cbn [value_all_cells] in Hcells. cbn [value_usable].
    rewrite (cell_within_not_dead rs stop l Hcells). reflexivity.
Qed.

(* A value that reaches symbols and locations of the heap only is not
   local (decision D46). *)
Lemma heap_value_not_local (rs : region_state) (v : value) :
  symbols_fresh rs = true ->
  value_for_ever rs v = true ->
  value_local rs v = false.
Proof.
  intros Hfresh. unfold value_for_ever. intros H.
  apply andb_true_iff in H as [Hheap Hcells]. destruct v; try reflexivity.
  - cbn [value_all] in Hheap.
    pose proof (heap_not_local rs s Hfresh Hheap) as H.
    unfold not_local_symbol in H. apply andb_true_iff in H as [_ H].
    destruct (value_local rs (Value_symbol s)); [ discriminate | reflexivity ].
  - cbn [value_all_cells] in Hcells. cbn [value_local].
    unfold heap_location in Hcells. unfold local_cell.
    destruct (lookup l (rs_cells rs)); [ discriminate | reflexivity ].
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
  intros (Hms & Hplaced & _ & Henv & _) Hin. unfold check_usable.
  destruct s as [x | sym | c]; cbn [lifetime_of eval_simple].
  - destruct (lookup x (cs_env cs)) as [l | ] eqn:Hl; [ | discriminate ].
    intros Husable Hv. specialize (Henv x l v Hl Hv).
    destruct l as [ | | r | ]; cbn [lifetime_usable lifetime_holds] in *.
    + exact (heap_within rs None v Henv).
    + exact (within_no_bound rs _ v Henv).
    + destruct (region_place_open _ _ ms None r Hms Husable)
        as (m & stop & Hplace & Hinm).
      rewrite Hplace in Henv.
      exact (within_no_bound rs _ v
               (Henv (marks_placed_in _ _ _ Hplaced Hinm))).
    + discriminate.
  - intros _ [= <-]. unfold value_within.
    cbn [value_all value_all_cells andb].
    unfold symbol_within. cbn [simple_below] in Hin.
    rewrite (below_base_heap rs sym Hin). reflexivity.
  - intros _ [= <-]. reflexivity.
Qed.

(* Decision D46, of the state that runs: a simple that the check takes
   to live for ever has a value that reaches symbols and locations of
   the heap only. *)
Lemma running_forever_heap (rs : region_state) (cs : check_state)
    (ms : list mark) (s : simple) (v : value) :
  running_checked rs cs ms ->
  simple_below (rs_base rs) s = true ->
  is_forever (lifetime_of cs s) = true ->
  eval_simple (state_vars (rs_state rs)) s = Some v ->
  value_for_ever rs v = true.
Proof.
  intros (_ & _ & _ & Henv & _) Hin.
  destruct s as [x | sym | c]; cbn [lifetime_of eval_simple].
  - destruct (lookup x (cs_env cs)) as [l | ] eqn:Hl; [ | discriminate ].
    intros Hforever Hv. specialize (Henv x l v Hl Hv).
    destruct l; cbn [is_forever lifetime_holds] in *; try discriminate.
    exact Henv.
  - intros _ [= <-]. unfold value_for_ever.
    cbn [value_all value_all_cells]. cbn [simple_below] in Hin.
    rewrite (below_base_heap rs sym Hin). reflexivity.
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
