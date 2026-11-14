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
   what a continuation is as its entry says to ([cont_ok]) is kept when the
   store grows and no frame is popped, when a region is ended, and when the
   user changes. The end of a region is by the order of the frames
   (FlambdaRegionsCallsSpec.frames_ordered). Where the store grows, a
   cell may be made or written: what is said of a value is kept if the
   heap binds every location that the value names, so the values that
   a continuation holds are asked to be such. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsReadsProofs
  FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsCellsProofs FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCallsProofs FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs.
Import ListNotations.

(* ======================================================================== *)
(* Marks                                                                    *)
(* ======================================================================== *)

Lemma marks_decrease_lt (m : mark) (ms : list mark) :
  marks_decrease (m :: ms) = true ->
  ∀ m' : mark, In m' ms -> (m' < m)%positive.
Proof.
  revert m. induction ms as [ | m1 ms IH]; intros m; [ contradiction | ].
  cbn [marks_decrease]. intros H. apply andb_true_iff in H as [Hlt Hrest].
  apply Pos.ltb_lt in Hlt. intros m' [<- | Hin]; [ exact Hlt | ].
  pose proof (IH m1 Hrest m' Hin). lia.
Qed.

Lemma marks_decrease_tail (m : mark) (ms : list mark) :
  marks_decrease (m :: ms) = true -> marks_decrease ms = true.
Proof.
  cbn [marks_decrease]. intros H. apply andb_true_iff in H as [_ H]. exact H.
Qed.

(* The place of an open region: its mark is one of the marks, and its
   bound is at most the bound of the newest and at most every mark
   above its own. *)
Lemma region_place_bound (opens : list variable) :
  ∀ (ms : list mark) (above : symbol) (r : variable) (m : mark)
    (stop : option symbol),
    region_place opens ms (Some above) r = Some (m, stop) ->
    (∀ m' : mark, In m' ms -> (m' < above)%positive) ->
    marks_decrease ms = true ->
    ∃ b : symbol,
      stop = Some b
      ∧ (b <= above)%positive
      ∧ In m ms
      ∧ ∀ m0 : mark, In m0 ms -> (m < m0)%positive -> (b <= m0)%positive.
Proof.
  induction opens as [ | x opens IH]; intros ms above r m stop;
    [ discriminate | ].
  destruct ms as [ | m1 ms]; [ discriminate | ]. cbn [region_place].
  destruct (Pos.eqb x r).
  - intros [= <- <-] Hbelow Hdecrease. exists above.
    split; [ reflexivity | ]. split; [ lia | ].
    split; [ left; reflexivity | ].
    intros m0 [<- | Hin] Hlt; [ lia | ].
    pose proof (marks_decrease_lt m1 ms Hdecrease m0 Hin). lia.
  - intros Hplace Hbelow Hdecrease.
    destruct (IH ms m1 r m stop Hplace (marks_decrease_lt m1 ms Hdecrease)
                (marks_decrease_tail m1 ms Hdecrease))
      as (b & -> & Hb & Hin & Hup).
    exists b. split; [ reflexivity | ].
    pose proof (Hbelow m1 (or_introl eq_refl)).
    split; [ lia | ]. split; [ right; exact Hin | ].
    intros m0 [<- | Hin0] Hlt; [ exact Hb | exact (Hup m0 Hin0 Hlt) ].
Qed.

(* The same with any bound of the newest: the mark is one of the
   marks. *)
Lemma region_place_in (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable) (m : mark)
    (stop : option symbol),
    region_place opens ms above r = Some (m, stop) -> In m ms.
Proof.
  induction opens as [ | x opens IH]; intros ms above r m stop;
    [ discriminate | ].
  destruct ms as [ | m1 ms]; [ discriminate | ]. cbn [region_place].
  destruct (Pos.eqb x r).
  - intros [= <- _]. left. reflexivity.
  - intros H. right. exact (IH _ _ _ _ _ H).
Qed.

(* The mark of the oldest open region is one of the marks, and at most
   each. *)
Lemma lowest_least (ms : list mark) :
  ∀ b : mark,
    lowest ms = Some b ->
    marks_decrease ms = true ->
    In b ms ∧ ∀ m0 : mark, In m0 ms -> (b <= m0)%positive.
Proof.
  induction ms as [ | m ms IH]; intros b; [ discriminate | ].
  destruct ms as [ | m1 ms].
  - intros [= <-] _. split; [ left; reflexivity | ].
    intros m0 [<- | []]. lia.
  - change (lowest (m :: m1 :: ms)) with (lowest (m1 :: ms)).
    intros Hlowest Hdecrease.
    destruct (IH b Hlowest (marks_decrease_tail _ _ Hdecrease)) as [Hin Hle].
    split; [ right; exact Hin | ].
    intros m0 [<- | Hin0]; [ | exact (Hle m0 Hin0) ].
    pose proof (marks_decrease_lt m (m1 :: ms) Hdecrease b Hin). lia.
Qed.

Lemma lowest_none (ms : list mark) : lowest ms = None -> ms = [].
Proof.
  induction ms as [ | m ms IH]; [ reflexivity | ].
  destruct ms as [ | m1 ms]; [ discriminate | ].
  change (lowest (m :: m1 :: ms)) with (lowest (m1 :: ms)).
  intros H. discriminate (IH H).
Qed.

(* The bound of what lives as long as the code, with a bound of the
   newest region above every mark. *)
Lemma caller_bound_least (ms : list mark) (above : symbol) :
  (∀ m' : mark, In m' ms -> (m' < above)%positive) ->
  marks_decrease ms = true ->
  ∃ b : symbol,
    caller_bound ms (Some above) = Some b
    ∧ (b <= above)%positive
    ∧ ∀ m0 : mark, In m0 ms -> (b <= m0)%positive.
Proof.
  intros Hbelow Hdecrease. unfold caller_bound.
  destruct (lowest ms) as [b | ] eqn:Hlowest.
  - destruct (lowest_least ms b Hlowest Hdecrease) as [Hin Hle].
    exists b. split; [ reflexivity | ].
    pose proof (Hbelow b Hin). split; [ lia | exact Hle ].
  - exists above. split; [ reflexivity | ]. split; [ lia | ].
    intros m0 Hin. rewrite (lowest_none ms Hlowest) in Hin. contradiction.
Qed.

(* ======================================================================== *)
(* The store grows                                                          *)
(* ======================================================================== *)

(* A value that lives for ever is as any lifetime says. *)
Lemma forever_holds (rs : region_state) (opens : list variable)
    (ms : list mark) (above : option symbol) (l : lifetime) (v : value) :
  value_for_ever rs v = true -> lifetime_holds rs opens ms above l v.
Proof.
  intros H. destruct l as [ | | r | ]; cbn [lifetime_holds].
  - exact H.
  - exact (heap_within rs _ v H).
  - destruct (region_place opens ms above r) as [[m stop] | ]; [ | exact I ].
    intros _. exact (heap_within rs _ v H).
  - exact I.
Qed.

(* The values that a continuation with extra arguments holds. *)
Lemma with_extra_values_all (P : value -> Prop) (c : cont_value)
    (extra : list value) :
  cont_values_all P (with_extra_args c extra) ->
  cont_values_all P c ∧ Forall P extra.
Proof.
  destruct extra as [ | v extra]; cbn [with_extra_args cont_values_all].
  - intros H. split; [ exact H | constructor ].
  - intros [Hextra Hc]. split; assumption.
Qed.

(* No frame is popped: a symbol of the heap stays one, a symbol that a
   frame binds stays bound, the marks drawn so far are on the stack
   as they were, and a mark that is on the stack stays. The cells
   that the heap of the first state binds are listed as they were;
   and a value that such a cell holds in the second state was in the
   cell, or lives for ever: the cell was written (decision D54). *)
Record store_grows (rs rs' : region_state) : Prop := Store_grows {
  grows_next : (rs_next rs <= rs_next rs')%positive;
  grows_heap :
    ∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true;
  grows_local :
    ∀ s : symbol,
      local_in (rs_frames rs) s = true -> local_in (rs_frames rs') s = true;
  grows_stack :
    ∀ m : mark,
      (m < rs_next rs)%positive ->
      on_stack m (rs_frames rs') = on_stack m (rs_frames rs);
  grows_on_stack :
    ∀ m : mark,
      on_stack m (rs_frames rs) = true -> on_stack m (rs_frames rs') = true;
  grows_cells : cells_same_on (state_heap (rs_state rs)) rs rs';
  grows_cell_values :
    ∀ (l : location) (c' : cell) (w : value),
      is_bound l (state_heap (rs_state rs)) = true ->
      lookup l (state_heap (rs_state rs')) = Some c' ->
      In w (values_of_cell c') ->
      (∃ c : cell,
         lookup l (state_heap (rs_state rs)) = Some c
         ∧ In w (values_of_cell c))
      ∨ value_for_ever rs' w = true
}.

Section Grows.

Variables rs rs' : region_state.
Hypothesis Hgrows : store_grows rs rs'.

(* The heap binds every location that the value names. *)
Definition closed (v : value) : Prop :=
  value_closed (state_heap (rs_state rs)) v.

Lemma grows_heap_value (v : value) :
  value_all (heap_symbol rs) v = true -> value_all (heap_symbol rs') v = true.
Proof. apply value_all_mono. exact (grows_heap rs rs' Hgrows). Qed.

Lemma grows_heap_location (l : location) :
  is_bound l (state_heap (rs_state rs)) = true ->
  heap_location rs l = true ->
  heap_location rs' l = true.
Proof.
  intros Hbound. unfold heap_location.
  rewrite (grows_cells rs rs' Hgrows l Hbound). exact (fun H => H).
Qed.

Lemma grows_cell_within (stop : option symbol) (l : location) :
  is_bound l (state_heap (rs_state rs)) = true ->
  cell_within rs stop l = true ->
  cell_within rs' stop l = true.
Proof.
  intros Hbound. unfold cell_within, heap_location.
  rewrite (grows_cells rs rs' Hgrows l Hbound).
  destruct (lookup l (rs_cells rs)) as [m | ]; cbn [is_some negb orb];
    [ | exact (fun H => H) ].
  intros H. apply andb_true_iff in H as [Hon Hbefore].
  rewrite (grows_on_stack rs rs' Hgrows m Hon), Hbefore. reflexivity.
Qed.

Lemma grows_for_ever (v : value) :
  closed v -> value_for_ever rs v = true -> value_for_ever rs' v = true.
Proof.
  intros Hclosed. unfold value_for_ever. intros H.
  apply andb_true_iff in H as [Hsymbols Hcells]. apply andb_true_iff. split.
  - exact (grows_heap_value v Hsymbols).
  - apply value_all_cells_iff. intros l Hl.
    exact
      (grows_heap_location l (Hclosed l Hl)
         (proj1 (value_all_cells_iff _ v) Hcells l Hl)).
Qed.

Lemma grows_within (stop : option symbol) (v : value) :
  closed v ->
  value_within rs stop v = true ->
  value_within rs' stop v = true.
Proof.
  intros Hclosed. unfold value_within. intros H.
  apply andb_true_iff in H as [Hsymbols Hcells]. apply andb_true_iff. split.
  - revert Hsymbols. apply value_all_mono. intros s. unfold symbol_within.
    intros H. apply orb_true_iff in H as [H | H].
    + rewrite (grows_heap rs rs' Hgrows s H). reflexivity.
    + apply andb_true_iff in H as [Hlocal Hbefore].
      rewrite (grows_local rs rs' Hgrows s Hlocal), Hbefore. apply orb_true_r.
  - apply value_all_cells_iff. intros l Hl.
    exact
      (grows_cell_within stop l (Hclosed l Hl)
         (proj1 (value_all_cells_iff _ v) Hcells l Hl)).
Qed.

Lemma grows_lifetime (opens : list variable) (ms : list mark)
    (above : option symbol) (l : lifetime) (v : value) :
  closed v ->
  (∀ m : mark, In m ms -> (m < rs_next rs)%positive) ->
  lifetime_holds rs opens ms above l v ->
  lifetime_holds rs' opens ms above l v.
Proof.
  intros Hclosed Hms. destruct l as [ | | r | ]; cbn [lifetime_holds].
  - exact (grows_for_ever v Hclosed).
  - exact (grows_within _ v Hclosed).
  - destruct (region_place opens ms above r) as [[m stop] | ] eqn:Hplace;
      [ | exact (fun H => H) ].
    intros H Hon. apply (grows_within _ v Hclosed). apply H.
    rewrite <- (grows_stack rs rs' Hgrows m
                  (Hms m (region_place_in _ _ _ _ _ _ Hplace))).
    exact Hon.
  - exact (fun H => H).
Qed.

(* What is read from a value: a cell that was there holds what it
   held, or what a store put in it, which lives for ever. *)
Lemma grows_contents (opens : list variable) (ms : list mark)
    (above : option symbol) (l : lifetime) (v : value) :
  heap_values_all closed (state_heap (rs_state rs)) ->
  closed v ->
  (∀ m : mark, In m ms -> (m < rs_next rs)%positive) ->
  contents_hold rs opens ms above l v ->
  contents_hold rs' opens ms above l v.
Proof.
  intros Hheap Hclosed Hms.
  destruct v; try exact (grows_lifetime opens ms above l _ Hclosed Hms).
  cbn [contents_hold]. intros H c' w Hl' Hw.
  destruct
    (grows_cell_values rs rs' Hgrows l0 c' w
       (Hclosed l0 (or_introl eq_refl)) Hl' Hw)
    as [(c & Hl & Hin) | Hforever].
  - apply (grows_lifetime opens ms above l w); [ | exact Hms | ].
    + exact (Hheap l0 c w (lookup_in _ _ _ Hl) Hin).
    + exact (H c w Hl Hin).
  - exact (forever_holds rs' opens ms above l w Hforever).
Qed.
Lemma grows_placed (ms : list mark) :
  (∀ m : mark, In m ms -> (m < rs_next rs)%positive) ->
  marks_placed (rs_frames rs) ms = true ->
  marks_placed (rs_frames rs') ms = true.
Proof.
  intros Hms. unfold marks_placed. intros H.
  apply andb_true_iff in H as [Hon Hdecrease].
  apply andb_true_iff. split; [ | exact Hdecrease ].
  apply forallb_forall. intros m Hin.
  rewrite (grows_stack rs rs' Hgrows m (Hms m Hin)).
  exact (forallb_in _ _ m Hon Hin).
Qed.

Variable cert : certificate.
Variable codes : list (code_id * code).

Lemma forallb_ltb_in (next : symbol) (ms : list mark) (m : mark) :
  forallb (fun m => Pos.ltb m next) ms = true -> In m ms ->
  (m < next)%positive.
Proof.
  intros H Hin. apply Pos.ltb_lt. exact (forallb_in _ _ m H Hin).
Qed.

Lemma grows_made_at (u : user) (vars : env)
    (recursive : list (continuation * continuation_handler))
    (cs : check_state) (base : list (continuation * cont_entry))
    (ms : list mark) (next : symbol) :
  heap_values_all closed (state_heap (rs_state rs)) ->
  env_values_all closed vars ->
  (u_limit u <= rs_next rs)%positive ->
  made_at cert codes rs u vars recursive cs base ms next ->
  made_at cert codes rs' u vars recursive cs base ms next.
Proof.
  intros Hheap Hvars Hlimit
    (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11 & H12).
  assert (Hms : ∀ m : mark, In m ms -> (m < rs_next rs)%positive).
  { intros m Hin. pose proof (forallb_ltb_in next ms m H7 Hin).
    apply Pos.leb_le in H8. lia. }
  repeat (split; [ assumption | ]).
  split; [ | split; [ | split; assumption ] ].
  - intros x l v Hl Hv.
    exact
      (grows_lifetime _ _ _ l v (env_values_all_lookup _ vars x v Hvars Hv)
         Hms (H9 x l v Hl Hv)).
  - intros x l v Hl Hv.
    exact
      (grows_contents _ _ _ l v Hheap
         (env_values_all_lookup _ vars x v Hvars Hv) Hms (H10 x l v Hl Hv)).
Qed.

Lemma grows_called_from (u : user) (a : caller) :
  (u_limit u <= rs_next rs)%positive ->
  called_from rs u a -> called_from rs' u a.
Proof.
  intros Hlimit (H1 & H2 & H3 & H4 & H5 & H6).
  split; [ exact H1 | ]. split; [ | repeat split; assumption ].
  apply grows_placed; [ | exact H2 ].
  intros m Hin. pose proof (forallb_ltb_in _ _ m H4 Hin).
  apply Pos.leb_le in H5. lia.
Qed.

Definition caller_bounded (a : caller) : Prop :=
  forallb (fun m => Pos.ltb m (caller_next a)) (caller_marks a) = true.

Lemma grows_caller_holds (a : caller) (l : lifetime) (v : value) :
  closed v ->
  caller_bounded a ->
  (caller_next a <= rs_next rs)%positive ->
  caller_holds rs a l v -> caller_holds rs' a l v.
Proof.
  intros Hclosed Hbounded Hnext [Husable Hholds]. split; [ exact Husable | ].
  apply grows_lifetime; [ exact Hclosed | | exact Hholds ].
  intros m Hin. pose proof (forallb_ltb_in _ _ m Hbounded Hin). lia.
Qed.

Lemma Forall2_impl_right {A B : Type} (R R' : A -> B -> Prop) (P : B -> Prop)
    (l : list A) (l' : list B) :
  (∀ (a : A) (b : B), P b -> R a b -> R' a b) ->
  Forall P l' ->
  Forall2 R l l' ->
  Forall2 R' l l'.
Proof.
  intros Himpl HP H. induction H as [ | a b l l' Hab _ IH]; [ constructor | ].
  inversion HP as [ | b0 l0 Hb Hl' ]; subst.
  constructor; [ exact (Himpl a b Hb Hab) | exact (IH Hl') ].
Qed.

(* The continuations, where the heap binds every location that the
   values they hold name. *)
Lemma grows_conts :
  heap_values_all closed (state_heap (rs_state rs)) ->
  (∀ (u : user) (ce : cont_entry) (c : cont_value),
     cont_ok cert codes rs u ce c ->
     cont_values_all closed c ->
     (u_limit u <= rs_next rs)%positive ->
     cont_ok cert codes rs' u ce c)
  ∧ (∀ (a : caller) (l : lifetime) (c : cont_value),
       ret_ok cert codes rs a l c ->
       cont_values_all closed c ->
       caller_bounded a ->
       (caller_next a <= rs_next rs)%positive ->
       ret_ok cert codes rs' a l c)
  ∧ (∀ (a : caller) (c : cont_value),
       raise_ok cert codes rs a c ->
       cont_values_all closed c ->
       caller_bounded a ->
       (caller_next a <= rs_next rs)%positive ->
       raise_ok cert codes rs' a c).
Proof.
  intros Hheap.
  apply (conts_ok_mutind cert codes rs
           (fun u ce c _ =>
              cont_values_all closed c ->
              (u_limit u <= rs_next rs)%positive ->
              cont_ok cert codes rs' u ce c)
           (fun a l c _ =>
              cont_values_all closed c ->
              caller_bounded a ->
              (caller_next a <= rs_next rs)%positive ->
              ret_ok cert codes rs' a l c)
           (fun a c _ =>
              cont_values_all closed c ->
              caller_bounded a ->
              (caller_next a <= rs_next rs)%positive ->
              raise_ok cert codes rs' a c)).
  - intros u params metadata handler entry vars conts recursive cs base ms
      next Hmade Hfits Hcheck Hincl Hconts IH Hnone [Hc_vars Hc_conts] Hlimit.
    apply (Ok_handler cert codes rs' u params metadata handler entry vars conts
             recursive cs base ms next);
      try assumption.
    + exact
        (grows_made_at u vars recursive cs base ms next Hheap Hc_vars Hlimit
           Hmade).
    + intros k ce Hk.
      apply (IH k ce Hk (resolve_values_all closed k conts Hc_conts)).
      cbn [u_limit].
      destruct Hmade as (_ & _ & _ & _ & _ & _ & _ & H8 & _).
      apply Pos.leb_le in H8. lia.
  - intros u c _ _. apply Ok_unreachable.
  - intros u local a c Hcalled Hret IH Hc Hlimit.
    apply (Ok_return cert codes rs' u local a c);
      [ exact (grows_called_from u a Hlimit Hcalled) | ].
    destruct Hcalled as (_ & _ & _ & H4 & H5 & _).
    apply IH; [ exact Hc | exact H4 | ]. apply Pos.leb_le in H5. lia.
  - intros u a c Hcalled Hraise IH Hc Hlimit.
    apply (Ok_raise cert codes rs' u a c);
      [ exact (grows_called_from u a Hlimit Hcalled) | ].
    destruct Hcalled as (_ & _ & _ & H4 & H5 & _).
    apply IH; [ exact Hc | exact H4 | ]. apply Pos.leb_le in H5. lia.
  - intros a l k c Hjump Hsome IH Houter Hc Hbounded Hnext.
    apply (Ret_to cert codes rs' a l k c Hjump); [ | exact Houter ].
    intros ce Hk. exact (IH ce Hk Hc Hnext).
  - intros a l l' ls remaining remaining_arity return_arity return_to exn_to
      Hholds Hunknown Hret IHret Hraise IHraise
      (Hc_remaining & Hc_return & Hc_exn) Hbounded Hnext.
    apply (Ret_over cert codes rs' a l l' ls remaining remaining_arity
             return_arity return_to exn_to);
      [ | exact Hunknown | exact (IHret Hc_return Hbounded Hnext)
        | exact (IHraise Hc_exn Hbounded Hnext) ].
    revert Hholds. apply Forall2_impl_right with (P := closed);
      [ | exact Hc_remaining ].
    intros l0 v Hv. exact (grows_caller_holds a l0 v Hv Hbounded Hnext).
  - intros a l l' c Hlasts Hret IH Hc Hbounded Hnext.
    exact (Ret_longer cert codes rs' a l l' c Hlasts (IH Hc Hbounded Hnext)).
  - intros a k simples extra c Heval Hholds Hjump Hsome IH Houter Hc Hbounded
      Hnext.
    destruct (with_extra_values_all closed c extra Hc) as [Hc_c Hc_extra].
    apply (Raise_to cert codes rs' a k simples extra c);
      [ exact Heval | | exact Hjump | | exact Houter ].
    + revert Hholds. apply Forall2_impl_right with (P := closed);
        [ | exact Hc_extra ].
      intros s v Hv. exact (grows_caller_holds a _ v Hv Hbounded Hnext).
    + intros ce Hk. exact (IH ce Hk Hc_c Hnext).
Qed.

End Grows.

(* ======================================================================== *)
(* A region is ended                                                        *)
(* ======================================================================== *)

(* A symbol that a frame binds and that is below the mark of [f] is
   bound by a frame below [f]. *)
Lemma end_keeps_local (above0 : list frame) (above : symbol) (f : frame)
    (below : list frame) (s : symbol) :
  frames_ordered_below above (above0 ++ f :: below) = true ->
  local_in (above0 ++ f :: below) s = true ->
  (s < frame_mark f)%positive ->
  local_in below s = true.
Proof.
  intros Hord Hlocal Hs. unfold local_in in *.
  replace (above0 ++ f :: below) with ((above0 ++ [f]) ++ below) in Hlocal
    by (rewrite <- app_assoc; reflexivity).
  rewrite live_locals_app, lookup_app in Hlocal.
  destruct (lookup s (live_locals (above0 ++ [f]))) as [allocated | ] eqn:Hl;
    [ | exact Hlocal ].
  pose proof (frames_ordered_cut _ _ _ _ _ _ Hord (lookup_in _ _ _ Hl)). lia.
Qed.

(* What is below a frame is in order below its mark, and the frames
   above it have greater marks. *)
Lemma frames_ordered_tail (above : list frame) :
  ∀ (top : symbol) (f : frame) (below : list frame),
    frames_ordered_below top (above ++ f :: below) = true ->
    frames_ordered_below (frame_mark f) below = true.
Proof.
  induction above as [ | g above IH]; intros top f below;
    cbn [app frames_ordered_below]; intros H;
    apply andb_true_iff in H as [_ H]; [ exact H | exact (IH _ _ _ H) ].
Qed.

Lemma frames_above_marks (above : list frame) :
  ∀ (top : symbol) (f : frame) (below : list frame) (g : frame),
    frames_ordered_below top (above ++ f :: below) = true ->
    In g above ->
    (frame_mark f < frame_mark g)%positive.
Proof.
  induction above as [ | g0 above IH]; intros top f below g;
    [ contradiction | ].
  cbn [app frames_ordered_below]. intros H [<- | Hin].
  - apply andb_true_iff in H as [_ H].
    assert (Hf : In f (above ++ f :: below))
      by (apply in_app_iff; right; left; reflexivity).
    exact (frames_ordered_below_marks _ _ f H Hf).
  - apply andb_true_iff in H as [_ H]. exact (IH _ _ _ _ H Hin).
Qed.

(* The frames down to that of the mark [m0] are popped; nothing else
   changes that a value is asked about: the cells are listed as they
   were, and the heap is the same. *)
Record store_cut (rs rs' : region_state) (m0 : mark) : Prop := Store_cut {
  cut_frames :
    ∃ (above : list frame) (f : frame),
      rs_frames rs = above ++ f :: rs_frames rs' ∧ frame_mark f = m0;
  cut_heap : ∀ s : symbol, heap_symbol rs' s = heap_symbol rs s;
  cut_cells : rs_cells rs' = rs_cells rs;
  cut_cell_values :
    state_heap (rs_state rs') = state_heap (rs_state rs)
}.

Section Cut.

Variables rs rs' : region_state.
Variable m0 : mark.
Hypothesis Hcut : store_cut rs rs' m0.
Hypothesis Hordered : frames_ordered rs = true.

Lemma cut_heap_value (v : value) :
  value_all (heap_symbol rs) v = true -> value_all (heap_symbol rs') v = true.
Proof.
  apply value_all_mono. intros s H. rewrite (cut_heap rs rs' m0 Hcut). exact H.
Qed.

Lemma cut_for_ever (v : value) :
  value_for_ever rs v = true -> value_for_ever rs' v = true.
Proof.
  unfold value_for_ever. intros H.
  apply andb_true_iff in H as [Hsymbols Hcells]. apply andb_true_iff. split.
  - exact (cut_heap_value v Hsymbols).
  - revert Hcells. apply value_all_cells_mono. intros l.
    unfold heap_location. rewrite (cut_cells rs rs' m0 Hcut).
    exact (fun H => H).
Qed.

(* A symbol that a frame binds and that was drawn before the mark is
   bound by a frame that stays. *)
Lemma cut_local (s : symbol) :
  local_in (rs_frames rs) s = true ->
  (s < m0)%positive ->
  local_in (rs_frames rs') s = true.
Proof.
  destruct (cut_frames rs rs' m0 Hcut) as (above & f & Hframes & <-).
  unfold frames_ordered in Hordered. rewrite Hframes in *.
  exact (end_keeps_local above _ f _ s Hordered).
Qed.

(* A mark that is on the stack after was on it before; one that was on
   it and was drawn before the mark stays. *)
Lemma cut_stack_back (m : mark) :
  on_stack m (rs_frames rs') = true -> on_stack m (rs_frames rs) = true.
Proof.
  destruct (cut_frames rs rs' m0 Hcut) as (above & f & -> & _).
  intros H. rewrite on_stack_app. apply orb_true_iff. right.
  unfold on_stack in *. cbn [existsb]. rewrite H. apply orb_true_r.
Qed.

Lemma cut_stack_below (m : mark) :
  on_stack m (rs_frames rs') = true -> (m < m0)%positive.
Proof.
  destruct (cut_frames rs rs' m0 Hcut) as (above & f & Hframes & <-).
  unfold frames_ordered in Hordered. rewrite Hframes in Hordered.
  intros Hon. destruct (on_stack_frame m _ Hon) as (g & Hin & <-).
  exact
    (frames_ordered_below_marks _ _ g
       (frames_ordered_tail above _ f _ Hordered) Hin).
Qed.

Lemma cut_stack_keeps (m : mark) :
  on_stack m (rs_frames rs) = true ->
  (m < m0)%positive ->
  on_stack m (rs_frames rs') = true.
Proof.
  destruct (cut_frames rs rs' m0 Hcut) as (above & f & Hframes & <-).
  unfold frames_ordered in Hordered. rewrite Hframes in *.
  rewrite on_stack_app. intros Hon Hlt.
  apply orb_true_iff in Hon as [Hon | Hon].
  - exfalso. destruct (on_stack_frame m _ Hon) as (g & Hin & <-).
    pose proof (frames_above_marks above _ f _ g Hordered Hin). lia.
  - unfold on_stack in *. cbn [existsb] in Hon.
    apply orb_true_iff in Hon as [Hon | Hon]; [ | exact Hon ].
    apply Pos.eqb_eq in Hon. lia.
Qed.

(* A value within a bound that is at most the mark: the symbols and
   the cells that it reaches in a frame are in frames that stay. *)
Lemma cut_within (b : symbol) (v : value) :
  (b <= m0)%positive ->
  value_within rs (Some b) v = true ->
  value_within rs' (Some b) v = true.
Proof.
  intros Hb. unfold value_within. intros H.
  apply andb_true_iff in H as [Hsymbols Hcells]. apply andb_true_iff. split.
  - revert Hsymbols. apply value_all_mono. intros s. unfold symbol_within.
    rewrite (cut_heap rs rs' m0 Hcut). intros H.
    apply orb_true_iff in H as [-> | H]; [ reflexivity | ].
    apply andb_true_iff in H as [Hlocal Hbefore]. cbn [before] in *.
    rewrite Hbefore, andb_true_r.
    apply Pos.ltb_lt in Hbefore.
    rewrite (cut_local s Hlocal); [ apply orb_true_r | lia ].
  - revert Hcells. apply value_all_cells_mono. intros l.
    unfold cell_within, heap_location. rewrite (cut_cells rs rs' m0 Hcut).
    destruct (lookup l (rs_cells rs)) as [m | ]; cbn [is_some negb orb];
      [ | exact (fun H => H) ].
    intros H. apply andb_true_iff in H as [Hon Hbefore]. cbn [before] in *.
    rewrite Hbefore, andb_true_r. apply Pos.ltb_lt in Hbefore.
    apply (cut_stack_keeps m Hon). lia.
Qed.

(* A lifetime of a state that was made or of a caller, whose bounds are
   at most the mark. *)
Lemma cut_lifetime (opens : list variable) (ms : list mark) (next : symbol)
    (l : lifetime) (v : value) :
  (∀ m : mark, In m ms -> (m < next)%positive) ->
  marks_decrease ms = true ->
  (In m0 ms ∨ (next <= m0)%positive) ->
  lifetime_holds rs opens ms (Some next) l v ->
  lifetime_holds rs' opens ms (Some next) l v.
Proof.
  intros Hms Hdecrease Hm0. destruct l as [ | | r | ]; cbn [lifetime_holds].
  - exact (cut_for_ever v).
  - destruct (caller_bound_least ms next Hms Hdecrease) as (b & -> & Hb & Hle).
    apply cut_within. destruct Hm0 as [Hin | Hnext];
      [ exact (Hle m0 Hin) | lia ].
  - destruct (region_place opens ms (Some next) r) as [[m stop] | ] eqn:Hplace;
      [ | exact (fun H => H) ].
    destruct (region_place_bound opens ms next r m stop Hplace Hms Hdecrease)
      as (b & -> & Hb & Hin & Hup).
    intros H Hon. apply cut_within; [ | exact (H (cut_stack_back m Hon)) ].
    destruct Hm0 as [Hin0 | Hnext]; [ | lia ].
    exact (Hup m0 Hin0 (cut_stack_below m Hon)).
  - exact (fun H => H).
Qed.

(* The same of what is read from a value: the heap is the same. *)
Lemma cut_contents (opens : list variable) (ms : list mark) (next : symbol)
    (l : lifetime) (v : value) :
  (∀ m : mark, In m ms -> (m < next)%positive) ->
  marks_decrease ms = true ->
  (In m0 ms ∨ (next <= m0)%positive) ->
  contents_hold rs opens ms (Some next) l v ->
  contents_hold rs' opens ms (Some next) l v.
Proof.
  intros Hms Hdecrease Hm0.
  destruct v; try exact (cut_lifetime opens ms next l _ Hms Hdecrease Hm0).
  cbn [contents_hold]. rewrite (cut_cell_values rs rs' m0 Hcut).
  intros H c w Hl Hw.
  exact (cut_lifetime opens ms next l w Hms Hdecrease Hm0 (H c w Hl Hw)).
Qed.

Variable cert : certificate.
Variable codes : list (code_id * code).

Lemma cut_made_at (u : user) (vars : env)
    (recursive : list (continuation * continuation_handler))
    (cs : check_state) (base : list (continuation * cont_entry))
    (ms : list mark) (next : symbol) :
  (In m0 ms ∨ (next <= m0)%positive) ->
  made_at cert codes rs u vars recursive cs base ms next ->
  made_at cert codes rs' u vars recursive cs base ms next.
Proof.
  intros Hm0
    (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11 & H12).
  repeat (split; [ assumption | ]).
  split; [ | split; [ | split; assumption ] ].
  - intros x l v Hl Hv.
    exact
      (cut_lifetime (cs_open cs) ms next l v
         (fun m => forallb_ltb_in next ms m H7)
         H6 Hm0 (H9 x l v Hl Hv)).
  - intros x l v Hl Hv.
    exact
      (cut_contents (cs_open cs) ms next l v
         (fun m => forallb_ltb_in next ms m H7)
         H6 Hm0 (H10 x l v Hl Hv)).
Qed.

(* The mark is one of the marks of the closure, or was drawn after the
   closure was made. *)
Lemma made_at_mark (u : user) (vars : env)
    (recursive : list (continuation * continuation_handler))
    (cs : check_state) (base : list (continuation * cont_entry))
    (ms : list mark) (next : symbol) :
  made_at cert codes rs u vars recursive cs base ms next ->
  (In m0 (u_marks u) ∨ (u_limit u <= m0)%positive) ->
  In m0 ms ∨ (next <= m0)%positive.
Proof.
  intros (_ & _ & _ & _ & _ & _ & _ & H8 & _ & _ & H11 & _) Hm0.
  apply Pos.leb_le in H8.
  destruct Hm0 as [Hin | Hlimit]; [ | right; lia ].
  destruct (Pos.ltb m0 next) eqn:Hlt;
    [ left; exact (H11 m0 Hin Hlt) | right; apply Pos.ltb_ge; exact Hlt ].
Qed.

Lemma cut_called_from (u : user) (a : caller) :
  (In m0 (u_marks u) ∨ (u_limit u <= m0)%positive) ->
  called_from rs u a ->
  called_from rs' u a ∧ (caller_next a <= m0)%positive.
Proof.
  intros Hm0 (H1 & H2 & H3 & H4 & H5 & H6).
  assert (Hnext : (caller_next a <= m0)%positive).
  { apply Pos.leb_le in H5. destruct Hm0 as [Hin | Hlimit]; [ | lia ].
    apply Pos.leb_le. exact (forallb_in _ _ m0 H6 Hin). }
  split; [ | exact Hnext ].
  split; [ exact H1 | ]. split; [ | repeat split; assumption ].
  unfold marks_placed in *. apply andb_true_iff in H2 as [Hon Hdecrease].
  apply andb_true_iff. split; [ | exact Hdecrease ].
  apply forallb_forall. intros m Hin.
  apply cut_stack_keeps; [ exact (forallb_in _ _ m Hon Hin) | ].
  pose proof (forallb_ltb_in _ _ m H4 Hin). lia.
Qed.

(* The marks of a caller decrease: they are placed. *)
Definition caller_decreasing (a : caller) : Prop :=
  marks_decrease (caller_marks a) = true.

Lemma cut_caller_holds (a : caller) (l : lifetime) (v : value) :
  caller_bounded a ->
  caller_decreasing a ->
  (caller_next a <= m0)%positive ->
  caller_holds rs a l v -> caller_holds rs' a l v.
Proof.
  intros Hbounded Hdecrease Hnext [Husable Hholds]. split; [ exact Husable | ].
  exact
    (cut_lifetime _ _ _ l v (fun m => forallb_ltb_in _ _ m Hbounded) Hdecrease
       (or_intror Hnext) Hholds).
Qed.

Lemma cut_conts :
  (∀ (u : user) (ce : cont_entry) (c : cont_value),
     cont_ok cert codes rs u ce c ->
     (In m0 (u_marks u) ∨ (u_limit u <= m0)%positive) ->
     cont_ok cert codes rs' u ce c)
  ∧ (∀ (a : caller) (l : lifetime) (c : cont_value),
       ret_ok cert codes rs a l c ->
       caller_bounded a ->
       caller_decreasing a ->
       (caller_next a <= m0)%positive ->
       ret_ok cert codes rs' a l c)
  ∧ (∀ (a : caller) (c : cont_value),
       raise_ok cert codes rs a c ->
       caller_bounded a ->
       caller_decreasing a ->
       (caller_next a <= m0)%positive ->
       raise_ok cert codes rs' a c).
Proof.
  apply (conts_ok_mutind cert codes rs
           (fun u ce c _ =>
              (In m0 (u_marks u) ∨ (u_limit u <= m0)%positive) ->
              cont_ok cert codes rs' u ce c)
           (fun a l c _ =>
              caller_bounded a ->
              caller_decreasing a ->
              (caller_next a <= m0)%positive ->
              ret_ok cert codes rs' a l c)
           (fun a c _ =>
              caller_bounded a ->
              caller_decreasing a ->
              (caller_next a <= m0)%positive ->
              raise_ok cert codes rs' a c)).
  - intros u params metadata handler entry vars conts recursive cs base ms
      next Hmade Hfits Hcheck Hincl Hconts IH Hnone Hm0.
    pose proof (made_at_mark u vars recursive cs base ms next Hmade Hm0)
      as Hm0'.
    apply (Ok_handler cert codes rs' u params metadata handler entry vars conts
             recursive cs base ms next);
      try assumption.
    + exact (cut_made_at u vars recursive cs base ms next Hm0' Hmade).
    + intros k ce Hk. exact (IH k ce Hk Hm0').
  - intros u c _. apply Ok_unreachable.
  - intros u local a c Hcalled Hret IH Hm0.
    destruct (cut_called_from u a Hm0 Hcalled) as [Hcalled' Hnext].
    apply (Ok_return cert codes rs' u local a c Hcalled').
    destruct Hcalled as (_ & H2 & _ & H4 & _).
    apply IH; [ exact H4 | | exact Hnext ].
    unfold marks_placed in H2. apply andb_true_iff in H2 as [_ H2]. exact H2.
  - intros u a c Hcalled Hraise IH Hm0.
    destruct (cut_called_from u a Hm0 Hcalled) as [Hcalled' Hnext].
    apply (Ok_raise cert codes rs' u a c Hcalled').
    destruct Hcalled as (_ & H2 & _ & H4 & _).
    apply IH; [ exact H4 | | exact Hnext ].
    unfold marks_placed in H2. apply andb_true_iff in H2 as [_ H2]. exact H2.
  - intros a l k c Hjump Hsome IH Houter Hbounded Hdecrease Hnext.
    apply (Ret_to cert codes rs' a l k c Hjump); [ | exact Houter ].
    intros ce Hk. exact (IH ce Hk (or_intror Hnext)).
  - intros a l l' ls remaining remaining_arity return_arity return_to exn_to
      Hholds Hunknown Hret IHret Hraise IHraise Hbounded Hdecrease Hnext.
    apply (Ret_over cert codes rs' a l l' ls remaining remaining_arity
             return_arity return_to exn_to);
      [ | exact Hunknown | exact (IHret Hbounded Hdecrease Hnext)
        | exact (IHraise Hbounded Hdecrease Hnext) ].
    revert Hholds. apply Forall2_impl. intros l0 v.
    exact (cut_caller_holds a l0 v Hbounded Hdecrease Hnext).
  - intros a l l' c Hlasts Hret IH Hbounded Hdecrease Hnext.
    exact
      (Ret_longer cert codes rs' a l l' c Hlasts (IH Hbounded Hdecrease Hnext)).
  - intros a k simples extra c Heval Hholds Hjump Hsome IH Houter Hbounded
      Hdecrease Hnext.
    apply (Raise_to cert codes rs' a k simples extra c);
      [ exact Heval | | exact Hjump | | exact Houter ].
    + revert Hholds. apply Forall2_impl. intros s v.
      exact (cut_caller_holds a _ v Hbounded Hdecrease Hnext).
    + intros ce Hk. exact (IH ce Hk (or_intror Hnext)).
Qed.

End Cut.

(* ======================================================================== *)
(* The user changes                                                         *)
(* ======================================================================== *)

(* A later user: the variables of regions that the first has are bound
   as they were; each of its marks is one of the first's or was drawn
   at or after the first's counter; its counter is no earlier. *)
Definition user_extends (u u' : user) : Prop :=
  (∀ r : variable,
     among r (u_regions u) = true ->
     lookup r (u_vars u') = lookup r (u_vars u)
     ∧ among r (u_regions u') = true)
  ∧ (∀ m : mark,
       In m (u_marks u') -> In m (u_marks u) ∨ (u_limit u <= m)%positive)
  ∧ (u_limit u <= u_limit u')%positive.

Lemma user_conts (cert : certificate) (codes : list (code_id * code))
    (rs : region_state) (u u' : user) (ce : cont_entry) (c : cont_value) :
  user_extends u u' ->
  cont_ok cert codes rs u ce c ->
  cont_ok cert codes rs u' ce c.
Proof.
  intros (Hregions & Hmarks & Hlimit) Hok.
  destruct Hok
    as [u params metadata handler entry vars conts recursive cs base ms next
          Hmade Hfits Hcheck Hincl Hconts Hnone
       | u c | u local a c Hcalled Hret | u a c Hcalled Hraise].
  - apply (Ok_handler cert codes rs u' params metadata handler entry vars conts
             recursive cs base ms next);
      try assumption.
    destruct Hmade
      as (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11 & H12).
    apply Pos.leb_le in H8.
    repeat (split; [ assumption | ]).
    split; [ apply Pos.leb_le; lia | ]. split; [ exact H9 | ].
    split; [ exact H10 | ]. split.
    + intros m Hin Hlt. apply Pos.ltb_lt in Hlt.
      destruct (Hmarks m Hin) as [Hin' | Hle]; [ | lia ].
      apply (H11 m Hin'). apply Pos.ltb_lt. exact Hlt.
    + intros r Hr. destruct (H12 r Hr) as [Hlookup Hamong].
      destruct (Hregions r Hamong) as [Hlookup' Hamong'].
      split; [ rewrite Hlookup'; exact Hlookup | exact Hamong' ].
  - apply Ok_unreachable.
  - apply (Ok_return cert codes rs u' local a c); [ | exact Hret ].
    destruct Hcalled as (H1 & H2 & H3 & H4 & H5 & H6).
    apply Pos.leb_le in H5.
    repeat (split; [ assumption | ]).
    split; [ apply Pos.leb_le; lia | ].
    apply forallb_forall. intros m Hin. apply Pos.leb_le.
    destruct (Hmarks m Hin) as [Hin' | Hle]; [ | lia ].
    apply Pos.leb_le. exact (forallb_in _ _ m H6 Hin').
  - apply (Ok_raise cert codes rs u' a c); [ | exact Hraise ].
    destruct Hcalled as (H1 & H2 & H3 & H4 & H5 & H6).
    apply Pos.leb_le in H5.
    repeat (split; [ assumption | ]).
    split; [ apply Pos.leb_le; lia | ].
    apply forallb_forall. intros m Hin. apply Pos.leb_le.
    destruct (Hmarks m Hin) as [Hin' | Hle]; [ | lia ].
    apply Pos.leb_le. exact (forallb_in _ _ m H6 Hin').
Qed.

(* ======================================================================== *)
(* What a cell holds lives as long as its location                          *)
(* ======================================================================== *)

(* What frames in order bind was drawn before what is above them. *)
Lemma frames_ordered_locals (frames : list frame) :
  ∀ top s : symbol,
    frames_ordered_below top frames = true ->
    local_in frames s = true ->
    (s < top)%positive.
Proof.
  induction frames as [ | f below IH]; intros top s;
    [ intros _ H; unfold local_in in H; discriminate H | ].
  cbn [frames_ordered_below]. rewrite local_in_cons. intros H Hs.
  apply andb_true_iff in H as [H Hbelow].
  apply andb_true_iff in H as [Hm Hlocals]. apply Pos.ltb_lt in Hm.
  apply orb_true_iff in Hs as [Hs | Hs].
  - destruct (lookup s (frame_locals f)) as [a | ] eqn:Hl; [ | discriminate ].
    pose proof (forallb_in _ _ (s, a) Hlocals (lookup_in _ _ _ Hl)) as H.
    cbv beta iota in H. apply andb_true_iff in H as [_ H].
    apply Pos.ltb_lt. exact H.
  - pose proof (IH _ s Hbelow Hs). lia.
Qed.

(* The frames below some others are in order as the whole is, and
   below the mark of each of the others. *)
Lemma frames_ordered_below_drop (above : list frame) :
  ∀ (top : symbol) (fr : list frame),
    frames_ordered_below top (above ++ fr) = true ->
    frames_ordered_below top fr = true.
Proof.
  induction above as [ | g above IH]; intros top fr H; [ exact H | ].
  cbn [app frames_ordered_below] in H.
  apply andb_true_iff in H as [H Hbelow].
  apply andb_true_iff in H as [Hm _]. apply Pos.ltb_lt in Hm.
  apply (frames_ordered_below_le fr (frame_mark g) top); [ lia | ].
  exact (IH _ fr Hbelow).
Qed.

Lemma frames_ordered_under (above : list frame) :
  ∀ (top : symbol) (g : frame) (fr : list frame),
    frames_ordered_below top (above ++ fr) = true ->
    In g above ->
    frames_ordered_below (frame_mark g) fr = true.
Proof.
  induction above as [ | g0 above IH]; intros top g fr H Hin;
    [ contradiction | ].
  cbn [app frames_ordered_below] in H.
  apply andb_true_iff in H as [_ Hbelow].
  destruct Hin as [<- | Hin].
  - exact (frames_ordered_below_drop above _ fr Hbelow).
  - exact (IH _ g fr Hbelow Hin).
Qed.

Lemma frames_from_head (m : mark) (frames fr : list frame) :
  frames_from m frames = Some fr ->
  ∃ (f : frame) (below : list frame), fr = f :: below ∧ frame_mark f = m.
Proof.
  induction frames as [ | f below IH]; cbn [frames_from]; [ discriminate | ].
  destruct (Pos.eqb (frame_mark f) m) eqn:Hm; [ | exact IH ].
  intros [= <-]. exists f, below.
  split; [ reflexivity | apply Pos.eqb_eq; exact Hm ].
Qed.

(* The frames from that of a mark down: what they bind, and their
   marks, were drawn before every mark that is on the stack and was
   drawn after that one. *)
Lemma frames_from_below_mark (rs : region_state) (m b : mark)
    (fr : list frame) :
  frames_ordered rs = true ->
  frames_from m (rs_frames rs) = Some fr ->
  on_stack b (rs_frames rs) = true ->
  (m < b)%positive ->
  (∀ s : symbol, local_in fr s = true -> (s < b)%positive)
  ∧ (∀ m' : mark, on_stack m' fr = true -> (m' < b)%positive).
Proof.
  intros Hord Hfr Hon Hlt. unfold frames_ordered in Hord.
  destruct (frames_from_app m _ fr Hfr) as (above & Hframes & _ & _).
  destruct (frames_from_head m _ fr Hfr) as (f & below & -> & Hf).
  rewrite Hframes in Hord, Hon. rewrite on_stack_app in Hon.
  assert (Hunder : frames_ordered_below b (f :: below) = true).
  { apply orb_true_iff in Hon as [Hon | Hon].
    - destruct (on_stack_frame b above Hon) as (g & Hin & <-).
      exact (frames_ordered_under above _ g _ Hord Hin).
    - exfalso.
      pose proof (frames_ordered_below_drop above _ _ Hord) as Hfr_ord.
      cbn [frames_ordered_below] in Hfr_ord.
      apply andb_true_iff in Hfr_ord as [_ Hbelow].
      rewrite on_stack_cons in Hon. apply orb_true_iff in Hon as [Hon | Hon].
      + apply Pos.eqb_eq in Hon. lia.
      + destruct (on_stack_frame b below Hon) as (g & Hin & <-).
        pose proof (frames_ordered_below_marks below _ g Hbelow Hin). lia. }
  split.
  - intros s Hs. exact (frames_ordered_locals _ b s Hunder Hs).
  - intros m' Hm'. destruct (on_stack_frame m' _ Hm') as (g & Hin & <-).
    exact (frames_ordered_below_marks _ b g Hunder Hin).
Qed.

(* What a cell holds is within every bound that its location is
   within, where the bound is a mark that is on the stack: the cell
   holds what is in the frame that it was made in or in a frame below
   it, or lives for ever. *)
Lemma cell_within_deep (rs : region_state) (stop : option symbol)
    (l : location) (c : cell) (w : value) :
  frames_ordered rs = true ->
  (∀ b : symbol, stop = Some b -> on_stack b (rs_frames rs) = true) ->
  cell_deep rs l c = true ->
  In w (values_of_cell c) ->
  cell_within rs stop l = true ->
  value_within rs stop w = true.
Proof.
  intros Hord Hstop Hdeep Hw. unfold cell_within, heap_location.
  unfold cell_deep in Hdeep.
  destruct (lookup l (rs_cells rs)) as [m | ] eqn:Hl; cbn [is_some negb orb].
  - intros H. apply andb_true_iff in H as [Hon Hbefore].
    destruct (on_stack_frames_from m _ Hon) as (fr & Hfr).
    rewrite Hfr in Hdeep.
    pose proof (forallb_in _ _ w Hdeep Hw) as H. cbv beta in H.
    apply andb_true_iff in H as [Hsymbols Hcells].
    destruct (frames_from_app m _ fr Hfr) as (above & Hframes & _ & _).
    assert (Hbelow :
      ∀ b : symbol,
        stop = Some b ->
        (∀ s : symbol, local_in fr s = true -> (s < b)%positive)
        ∧ (∀ m' : mark, on_stack m' fr = true -> (m' < b)%positive)).
    { intros b ->. cbn [before] in Hbefore. apply Pos.ltb_lt in Hbefore.
      exact (frames_from_below_mark rs m b fr Hord Hfr (Hstop b eq_refl)
               Hbefore). }
    unfold value_within. apply andb_true_iff. split.
    + revert Hsymbols. apply value_all_mono. intros s Hs.
      unfold symbol_within.
      apply orb_true_iff in Hs as [-> | Hs]; [ reflexivity | ].
      apply orb_true_iff. right. apply andb_true_iff. split.
      * rewrite Hframes. exact (local_in_below above fr s Hs).
      * destruct stop as [b | ]; [ | reflexivity ]. cbn [before].
        apply Pos.ltb_lt. exact (proj1 (Hbelow b eq_refl) s Hs).
    + revert Hcells. apply value_all_cells_mono. intros l'.
      unfold cell_within, cell_in. intros H.
      apply orb_true_iff in H as [-> | H]; [ reflexivity | ].
      apply orb_true_iff. right.
      destruct (lookup l' (rs_cells rs)) as [m' | ]; [ | discriminate ].
      apply andb_true_iff. split.
      * rewrite Hframes, on_stack_app, H. apply orb_true_r.
      * destruct stop as [b | ]; [ | reflexivity ]. cbn [before].
        apply Pos.ltb_lt. exact (proj2 (Hbelow b eq_refl) m' H).
  - intros _. exact (heap_within rs stop w (forallb_in _ _ w Hdeep Hw)).
Qed.

(* The bound of an open region is the bound of the newest, or one of
   the marks. *)
Lemma region_place_stop (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable) (m : mark)
    (b : symbol),
    region_place opens ms above r = Some (m, Some b) ->
    above = Some b ∨ In b ms.
Proof.
  induction opens as [ | x opens IH]; intros ms above r m b;
    [ discriminate | ].
  destruct ms as [ | m1 ms]; [ discriminate | ]. cbn [region_place].
  destruct (Pos.eqb x r).
  - intros [= _ ->]. left. reflexivity.
  - intros H. right.
    destruct (IH ms (Some m1) r m b H) as [[= <-] | Hin];
      [ left; reflexivity | right; exact Hin ].
Qed.

(* In the state that runs, what is read from a value is as the
   lifetime of the value says: the bounds are marks of open regions,
   which are on the stack. *)
Lemma holds_contents (rs : region_state) (opens : list variable)
    (ms : list mark) (l : lifetime) (v : value) :
  frames_ordered rs = true ->
  cells_deep rs = true ->
  marks_placed (rs_frames rs) ms = true ->
  lifetime_holds rs opens ms None l v ->
  contents_hold rs opens ms None l v.
Proof.
  intros Hord Hdeep Hplaced. destruct v; try exact (fun H => H).
  cbn [contents_hold]. intros H c w Hl Hw.
  assert (Hcell : cell_deep rs l0 c = true).
  { unfold cells_deep in Hdeep. apply andb_true_iff in Hdeep as [_ Hdeep].
    exact (forallb_in _ _ (l0, c) Hdeep (lookup_in _ _ _ Hl)). }
  assert (Hdecrease : marks_decrease ms = true).
  { unfold marks_placed in Hplaced.
    apply andb_true_iff in Hplaced as [_ Hplaced]. exact Hplaced. }
  destruct l as [ | | r | ]; cbn [lifetime_holds] in *.
  - unfold value_for_ever in H. cbn [value_all value_all_cells andb] in H.
    unfold heap_location in H. unfold cell_deep in Hcell.
    destruct (lookup l0 (rs_cells rs)); [ discriminate | ].
    exact (forallb_in _ _ w Hcell Hw).
  - apply (cell_within_deep rs _ l0 c w Hord); [ | exact Hcell | exact Hw | ].
    + intros b Hb. unfold caller_bound in Hb.
      destruct (lowest ms) as [b0 | ] eqn:Hlowest; [ | discriminate ].
      injection Hb as <-.
      exact
        (marks_placed_in _ ms b0 Hplaced
           (proj1 (lowest_least ms b0 Hlowest Hdecrease))).
    + unfold value_within in H. cbn [value_all value_all_cells andb] in H.
      exact H.
  - destruct (region_place opens ms None r) as [[m stop] | ] eqn:Hplace;
      [ | exact I ].
    intros Hon.
    apply (cell_within_deep rs stop l0 c w Hord);
      [ | exact Hcell | exact Hw | ].
    + intros b ->.
      destruct (region_place_stop opens ms None r m b Hplace) as [Hb | Hb];
        [ discriminate | exact (marks_placed_in _ _ b Hplaced Hb) ].
    + specialize (H Hon). unfold value_within in H.
      cbn [value_all value_all_cells andb] in H. exact H.
  - exact I.
Qed.
