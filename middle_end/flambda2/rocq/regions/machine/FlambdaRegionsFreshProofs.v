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

(* Lemmas for the proofs of the statements of
   FlambdaRegionsInvariantSpec.v: the first part of the invariant,
   [symbols_fresh]. The state that a run starts in has it, a step keeps
   it, and with it what is gone stays gone. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsReadsProofs
  FlambdaRegionsStepsProofs.
From OxCamlFlambda2 Require FlambdaHeapProofs.
Import ListNotations.

(* ======================================================================== *)
(* A greater counter                                                        *)
(* ======================================================================== *)

Lemma symbol_below_mono (next next' s : symbol) :
  (next <= next')%positive ->
  symbol_below next s = true ->
  symbol_below next' s = true.
Proof. unfold symbol_below. rewrite !Pos.ltb_lt. lia. Qed.

Lemma forallb_Forall_impl {A : Type} (f g : A -> bool) (l : list A) :
  Forall (fun a => f a = true -> g a = true) l ->
  forallb f l = true ->
  forallb g l = true.
Proof.
  intros Hall. apply forallb_impl. intros a Hin.
  exact (proj1 (Forall_forall _ l) Hall a Hin).
Qed.

Lemma forallb_pairs_Forall_impl {A B : Type} (f g : B -> bool)
    (l : list (A * B)) :
  Forall (fun ab => f (snd ab) = true -> g (snd ab) = true) l ->
  forallb (fun '(_, b) => f b) l = true ->
  forallb (fun '(_, b) => g b) l = true.
Proof.
  rewrite !forallb_pairs. intros Hall. apply forallb_impl. intros b Hin.
  apply in_map_iff in Hin as ([a b'] & <- & Hin).
  exact (proj1 (Forall_forall _ l) Hall (a, b') Hin).
Qed.

Lemma value_below_mono (next next' : symbol) (v : value) :
  (next <= next')%positive ->
  value_below next v = true ->
  value_below next' v = true.
Proof.
  intros Hle.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args arity IHcallee IHargs]
    using FlambdaHeapProofs.value_induction;
    simpl.
  - auto.
  - exact (forallb_pairs_Forall_impl (value_below next) (value_below next')
             slots IH).
  - exact (forallb_pairs_Forall_impl (value_below next) (value_below next')
             slots IH).
  - exact (symbol_below_mono next next' s Hle).
  - exact (forallb_Forall_impl _ _ fields IH).
  - exact (forallb_Forall_impl _ _ elements IH).
  - auto.
  - auto.
  - intros H. apply andb_true_iff in H as [Hcallee Hargs].
    rewrite (IHcallee Hcallee).
    exact (forallb_Forall_impl _ _ args IHargs Hargs).
Qed.

Lemma allocated_below_mono (next next' : symbol)
    (allocated : statically_allocated_value) :
  (next <= next')%positive ->
  allocated_below next allocated = true ->
  allocated_below next' allocated = true.
Proof.
  intros Hle. rewrite !allocated_below_parts. intros H.
  apply andb_true_iff in H as [Hvalues Hsymbols]. apply andb_true_iff. split.
  - revert Hvalues. apply forallb_impl. intros v _.
    exact (value_below_mono next next' v Hle).
  - revert Hsymbols. apply forallb_impl. intros s _.
    exact (symbol_below_mono next next' s Hle).
Qed.

(* ======================================================================== *)
(* The simples of a term that names no symbol drawn                         *)
(* ======================================================================== *)

Lemma eval_simple_below (base next : symbol) (vars : env) (s : simple)
    (v : value) :
  (base <= next)%positive ->
  simple_below base s = true ->
  env_values_all (fun v => value_below next v = true) vars ->
  eval_simple vars s = Some v ->
  value_below next v = true.
Proof.
  intros Hle Hs Hvars. destruct s as [x | sym | c]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. exact (symbol_below_mono base next sym Hle Hs).
  - intros [= <-]. reflexivity.
Qed.

Lemma evaluated_simples_below (base : symbol) (e : expr) (sim : simple) :
  expr_below base e = true ->
  In sim (evaluated_simples e) ->
  simple_below base sim = true.
Proof.
  destruct e
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - intros He Hin. apply evaluated_simples_let in Hin. simpl in He.
    apply andb_true_iff in He as [Hd _]. exact (forallb_in _ _ sim Hd Hin).
  - intros He Hin. simpl in He, Hin. apply andb_true_iff in He as [Hb _].
    apply in_flat_map in Hin as (binding & Hbinding & Hin).
    pose proof (forallb_in _ _ binding Hb Hbinding) as Hbelow.
    destruct binding as [id c | id | s const]; simpl in Hin;
      try contradiction.
    simpl in Hbelow. apply andb_true_iff in Hbelow as [_ Hconst].
    apply andb_true_iff in Hconst as [Hsimples _].
    exact (forallb_in _ _ sim Hsimples Hin).
  - intros _ [].
  - intros _ [].
  - intros He Hin. exact (forallb_in _ _ sim He Hin).
  - intros He Hin. exact (forallb_in _ _ sim He Hin).
  - intros He Hin. simpl in He, Hin. apply andb_true_iff in He as [Hs _].
    destruct Hin as [<- | []]. exact Hs.
  - intros _ [].
Qed.

Lemma named_simples_below (base next : symbol) (vars : env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) (sim : simple)
    (w : value) :
  (base <= next)%positive ->
  expr_below base (Let x kind d body) = true ->
  env_values_all (fun v => value_below next v = true) vars ->
  In sim (simples_of_named d) ->
  eval_simple vars sim = Some w ->
  value_below next w = true.
Proof.
  intros Hle He Hvars Hin. simpl in He. apply andb_true_iff in He as [Hd _].
  exact
    (eval_simple_below base next vars sim w Hle (forallb_in _ _ sim Hd Hin)
       Hvars).
Qed.

Lemma eval_static_const_below (base next : symbol) (vars : env)
    (const : static_const) (allocated : statically_allocated_value) :
  (base <= next)%positive ->
  static_const_below base const = true ->
  env_values_all (fun v => value_below next v = true) vars ->
  eval_static_const vars const = Some allocated ->
  allocated_below next allocated = true.
Proof.
  intros Hle Hconst Hvars Heval.
  apply andb_true_iff in Hconst as [Hsimples Hsymbols].
  destruct (eval_static_const_makes vars const allocated Heval)
    as [Hvalues Hsyms].
  rewrite allocated_below_parts. apply andb_true_iff. split.
  - apply forallb_forall. intros w Hin.
    destruct (Hvalues w Hin) as (sim & Hsim & Hw).
    exact
      (eval_simple_below base next vars sim w Hle
         (forallb_in _ _ sim Hsimples Hsim) Hvars Hw).
  - rewrite Hsyms.
    destruct const
      as [ | slot closure_symbols function_decls value_slots | | | | ];
      try reflexivity.
    rewrite <- forallb_pairs. revert Hsymbols. apply forallb_impl.
    intros [slot' s] _. exact (symbol_below_mono base next s Hle).
Qed.

(* ======================================================================== *)
(* The test of what is written                                              *)
(* ======================================================================== *)

Lemma expr_below_let (base : symbol) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  expr_below base (Let x kind d body) = true -> expr_below base body = true.
Proof. simpl. intros H. apply andb_true_iff in H as [_ H]. exact H. Qed.

Lemma expr_below_let_static (base : symbol) (bindings : list static_binding)
    (body : expr) :
  expr_below base (Let_static bindings body) = true ->
  expr_below base body = true
  ∧ ∀ (id : code_id) (c : code),
      In (Bind_code id c) bindings -> code_below base c = true.
Proof.
  simpl. intros H. apply andb_true_iff in H as [Hb H]. split; [ exact H | ].
  intros id c Hin. exact (forallb_in _ _ _ Hb Hin).
Qed.

Lemma expr_below_let_cont (base : symbol) (k : continuation)
    (h : continuation_handler) (body : expr) :
  expr_below base (Let_cont_non_recursive k h body) = true ->
  handler_below base h = true ∧ expr_below base body = true.
Proof. simpl. intros H. apply andb_true_iff in H. exact H. Qed.

Lemma expr_below_let_cont_rec (base : symbol)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  expr_below base (Let_cont_recursive handlers body) = true ->
  forallb (fun '(_, h) => handler_below base h) handlers = true
  ∧ expr_below base body = true.
Proof. simpl. intros H. apply andb_true_iff in H. exact H. Qed.

Lemma expr_below_switch (base : symbol) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (i : Z) (ac : apply_cont) :
  expr_below base (Switch scrutinee arms) = true ->
  find_arm i arms = Some ac ->
  expr_below base (Apply_cont ac) = true.
Proof.
  simpl. intros H Harm. apply andb_true_iff in H as [_ Harms].
  destruct (find_arm_some_in i arms ac Harm) as (i' & Hin).
  exact (forallb_in _ _ (i', ac) Harms Hin).
Qed.

Lemma handler_below_handler (base : symbol) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  handler_below base (Handler params metadata handler) = true ->
  expr_below base handler = true.
Proof. exact (fun H => H). Qed.

Lemma code_below_body (base : symbol) (c : code) :
  code_below base c = true -> expr_below base (code_body c) = true.
Proof. destruct c. exact (fun H => H). Qed.

(* What a state has written names no symbol at or above [base]: its
   expression, the handlers that its continuations hold, the code of
   its static environment. *)
Definition written_below (base : symbol) (s : state) : Prop :=
  written_ok (expr_below base) (handler_below base) (code_below base) s.

(* A step keeps it. *)
Lemma region_step_keeps_written_below (rs rs' : region_state) :
  written_below (rs_base rs) (rs_state rs) ->
  region_step rs = Region_step rs' ->
  written_below (rs_base rs') (rs_state rs').
Proof.
  intros Hw Hstep.
  rewrite (proj1 (region_step_draws_upwards rs rs' Hstep)).
  exact
    (region_step_written (expr_below (rs_base rs))
       (handler_below (rs_base rs)) (code_below (rs_base rs))
       (expr_below_let _) (expr_below_let_static _) (expr_below_let_cont _)
       (expr_below_let_cont_rec _) (expr_below_switch _)
       (handler_below_handler _) (code_below_body _) rs rs' Hw Hstep).
Qed.

(* ======================================================================== *)
(* The parts of the test                                                    *)
(* ======================================================================== *)

(* What a state holds names no symbol at or above [next]: the values of
   its variables and of its continuations. *)
Definition held_below (next : symbol) (s : state) : Prop :=
  vals_ok (fun v => value_below next v = true) s.

(* The symbols of a frame were drawn, and what it binds them to names
   no symbol at or above [next]. *)
Definition frame_drawn (base next : symbol) (f : frame) : Prop :=
  (base <= frame_mark f)%positive
  ∧ (frame_mark f < next)%positive
  ∧ ∀ (s : symbol) (allocated : statically_allocated_value),
      In (s, allocated) (frame_locals f) ->
      (base <= s)%positive
      ∧ (s < next)%positive
      ∧ allocated_below next allocated = true.

(* The symbols that are bound, in the static environment and in the
   frames: what the static environment binds is below [next]; the
   frames are of symbols drawn; the mark of a frame is bound to
   nothing; no symbol is bound in a frame and in the static
   environment. *)
Definition store_fresh (base next : symbol)
    (values : list (symbol * statically_allocated_value))
    (frames : list frame) : Prop :=
  (base <= next)%positive
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
      In (s, allocated) values ->
      (s < next)%positive ∧ allocated_below next allocated = true)
  ∧ (∀ f : frame, In f frames -> frame_drawn base next f)
  ∧ (∀ f : frame,
      In f frames ->
      lookup (frame_mark f) (live_locals frames ++ values) = None)
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
      In (s, allocated) (live_locals frames) -> lookup s values = None).

Lemma negb_is_some_none {A : Type} (o : option A) :
  negb (is_some o) = true <-> o = None.
Proof.
  destruct o; simpl; split; intros H; first [ reflexivity | discriminate ].
Qed.

Lemma symbol_between_iff (base next s : symbol) :
  symbol_between base next s = true
  <-> (base <= s)%positive ∧ (s < next)%positive.
Proof.
  unfold symbol_between. rewrite andb_true_iff, Pos.leb_le, Pos.ltb_lt.
  reflexivity.
Qed.

Lemma frame_fresh_drawn (base next : symbol) (f : frame) :
  frame_fresh base next f = true <-> frame_drawn base next f.
Proof.
  unfold frame_fresh, frame_drawn. split.
  - intros H. apply andb_true_iff in H as [Hmark Hlocals].
    apply symbol_between_iff in Hmark as [Hle Hlt].
    split; [ exact Hle | ]. split; [ exact Hlt | ].
    intros s allocated Hin.
    pose proof (forallb_in _ _ (s, allocated) Hlocals Hin) as H.
    cbv beta iota in H. apply andb_true_iff in H as [Hs Ha].
    apply symbol_between_iff in Hs as [H1 H2]. exact (conj H1 (conj H2 Ha)).
  - intros (Hle & Hlt & Hlocals). apply andb_true_iff. split.
    + apply symbol_between_iff. exact (conj Hle Hlt).
    + apply forallb_forall. intros [s allocated] Hin.
      destruct (Hlocals s allocated Hin) as (H1 & H2 & Ha).
      apply andb_true_iff.
      split; [ apply symbol_between_iff; exact (conj H1 H2) | exact Ha ].
Qed.

(* The test [symbols_fresh], in three parts: what is written, what is
   held, and the symbols that are bound. *)
Lemma symbols_fresh_parts (rs : region_state) :
  symbols_fresh rs = true
  <-> written_below (rs_base rs) (rs_state rs)
      ∧ held_below (rs_next rs) (rs_state rs)
      ∧ store_fresh (rs_base rs) (rs_next rs)
          (static_values (state_statics (rs_state rs))) (rs_frames rs).
Proof.
  unfold symbols_fresh. cbv zeta. split.
  - intros H.
    apply andb_true_iff in H as [H H9]. apply andb_true_iff in H as [H H8].
    apply andb_true_iff in H as [H H7]. apply andb_true_iff in H as [H H6].
    apply andb_true_iff in H as [H H5]. apply andb_true_iff in H as [H H4].
    apply andb_true_iff in H as [H H3]. apply andb_true_iff in H as [H1 H2].
    apply conts_below_split in H5 as [H5t H5v].
    split; [ exact (conj H2 (conj H5t H3)) | ].
    split; [ exact (conj (proj1 (forallb_pairs_Forall _ _) H4) H5v) | ].
    split; [ apply Pos.leb_le; exact H1 | ]. split.
    { intros s allocated Hin.
      pose proof (forallb_in _ _ (s, allocated) H6 Hin) as H.
      cbv beta iota in H. apply andb_true_iff in H as [Hs Ha].
      split; [ apply Pos.ltb_lt; exact Hs | exact Ha ]. }
    split.
    { intros f Hin. apply frame_fresh_drawn. exact (forallb_in _ _ f H7 Hin). }
    split.
    { intros f Hin. apply negb_is_some_none.
      exact (forallb_in _ _ f H8 Hin). }
    intros s allocated Hin. apply negb_is_some_none.
    exact (forallb_in _ _ (s, allocated) H9 Hin).
  - intros ((H2 & H5t & H3) & (H4 & H5v) & H1 & H6 & H7 & H8 & H9).
    repeat (apply andb_true_iff; split).
    + apply Pos.leb_le. exact H1.
    + exact H2.
    + exact H3.
    + exact (proj2 (forallb_pairs_Forall _ _) H4).
    + apply conts_below_split. exact (conj H5t H5v).
    + apply forallb_forall. intros [s allocated] Hin.
      destruct (H6 s allocated Hin) as [Hs Ha]. apply andb_true_iff.
      split; [ apply Pos.ltb_lt; exact Hs | exact Ha ].
    + apply forallb_forall. intros f Hin. apply frame_fresh_drawn.
      exact (H7 f Hin).
    + apply forallb_forall. intros f Hin. apply negb_is_some_none.
      exact (H8 f Hin).
    + apply forallb_forall. intros [s allocated] Hin.
      apply negb_is_some_none. exact (H9 s allocated Hin).
Qed.

(* ======================================================================== *)
(* The symbols that are bound                                               *)
(* ======================================================================== *)

Lemma frame_drawn_mono (base next next' : symbol) (f : frame) :
  (next <= next')%positive ->
  frame_drawn base next f ->
  frame_drawn base next' f.
Proof.
  intros Hle (Hm1 & Hm2 & Hlocals). split; [ exact Hm1 | ]. split; [ lia | ].
  intros s allocated Hin. destruct (Hlocals s allocated Hin) as (H1 & H2 & Ha).
  split; [ exact H1 | ].
  split; [ lia | exact (allocated_below_mono _ _ _ Hle Ha) ].
Qed.

(* What is bound, in a frame or in the static environment, is below the
   counter. *)
Lemma store_bound_below (base next : symbol)
    (values : list (symbol * statically_allocated_value))
    (frames : list frame) (s : symbol)
    (allocated : statically_allocated_value) :
  store_fresh base next values frames ->
  In (s, allocated) (live_locals frames ++ values) ->
  (s < next)%positive ∧ allocated_below next allocated = true.
Proof.
  intros (_ & Hvalues & Hframes & _ & _) Hin.
  apply in_app_or in Hin as [Hin | Hin]; [ | exact (Hvalues s allocated Hin) ].
  unfold live_locals in Hin. apply in_flat_map in Hin as (f & Hf & Hin).
  destruct (Hframes f Hf) as (_ & _ & Hlocals).
  destruct (Hlocals s allocated Hin) as (_ & Hs & Ha). exact (conj Hs Ha).
Qed.

Lemma store_lookup_above (base next : symbol)
    (values : list (symbol * statically_allocated_value))
    (frames : list frame) (s : symbol) :
  store_fresh base next values frames ->
  (next <= s)%positive ->
  lookup s (live_locals frames ++ values) = None.
Proof.
  intros Hstore Hle. apply lookup_not_in. intros allocated Hin.
  destruct (store_bound_below _ _ _ _ _ _ Hstore Hin) as [Hlt _]. lia.
Qed.

(* A frame is pushed, with the counter for its mark. *)
Lemma store_fresh_begin (base next : symbol)
    (values : list (symbol * statically_allocated_value))
    (frames : list frame) (kind : mark_kind) :
  store_fresh base next values frames ->
  store_fresh base (Pos.succ next) values (Frame next kind [] :: frames).
Proof.
  intros Hstore.
  pose proof Hstore as (Hle & Hvalues & Hframes & Hmarks & Hlocals).
  assert (Hsucc : (next <= Pos.succ next)%positive) by lia.
  split; [ lia | ]. split.
  { intros s allocated Hin. destruct (Hvalues s allocated Hin) as [Hs Ha].
    split; [ lia | exact (allocated_below_mono _ _ _ Hsucc Ha) ]. }
  split.
  { intros f [<- | Hin].
    - split; [ exact Hle | ]. split; [ simpl; lia | ].
      simpl. intros s allocated [].
    - exact (frame_drawn_mono base next (Pos.succ next) f Hsucc
               (Hframes f Hin)). }
  split.
  { intros f [<- | Hin].
    - exact (store_lookup_above _ _ _ _ next Hstore (Pos.le_refl next)).
    - exact (Hmarks f Hin). }
  exact Hlocals.
Qed.

(* Frames are popped. *)
Lemma store_fresh_end (base next : symbol)
    (values : list (symbol * statically_allocated_value))
    (above : list frame) (f : frame) (below : list frame) :
  store_fresh base next values (above ++ f :: below) ->
  store_fresh base next values below.
Proof.
  intros (Hle & Hvalues & Hframes & Hmarks & Hlocals).
  split; [ exact Hle | ]. split; [ exact Hvalues | ]. split.
  { intros f' Hin. apply Hframes. apply in_or_app. right. right. exact Hin. }
  split.
  { intros f' Hin.
    assert (Hin' : In f' (above ++ f :: below))
      by (apply in_or_app; right; right; exact Hin).
    pose proof (Hmarks f' Hin') as Hnone. rewrite live_locals_app in Hnone.
    apply lookup_app_none in Hnone as [Hnone Hv].
    apply lookup_app_none in Hnone as [_ Hnone].
    change (live_locals (f :: below))
      with (frame_locals f ++ live_locals below) in Hnone.
    apply lookup_app_none in Hnone as [_ Hnone].
    apply lookup_app_none. exact (conj Hnone Hv). }
  intros s allocated Hin. apply (Hlocals s allocated).
  rewrite live_locals_app. apply in_or_app. right.
  change (live_locals (f :: below)) with (frame_locals f ++ live_locals below).
  apply in_or_app. right. exact Hin.
Qed.

(* A group of static bindings is added, of symbols that were not
   drawn. *)
Lemma store_fresh_static (base next : symbol)
    (values : list (symbol * statically_allocated_value))
    (frames : list frame)
    (new_values : list (symbol * statically_allocated_value)) :
  store_fresh base next values frames ->
  (∀ (s : symbol) (allocated : statically_allocated_value),
    In (s, allocated) new_values ->
    (s < base)%positive ∧ allocated_below next allocated = true) ->
  store_fresh base next (new_values ++ values) frames.
Proof.
  intros (Hle & Hvalues & Hframes & Hmarks & Hlocals) Hnew.
  assert (Habove :
    ∀ s : symbol, (base <= s)%positive -> lookup s new_values = None).
  { intros s Hs. apply lookup_not_in. intros allocated Hin.
    destruct (Hnew s allocated Hin) as [Hlt _]. lia. }
  split; [ exact Hle | ]. split.
  { intros s allocated Hin. apply in_app_or in Hin as [Hin | Hin].
    - destruct (Hnew s allocated Hin) as [Hs Ha]. split; [ lia | exact Ha ].
    - exact (Hvalues s allocated Hin). }
  split; [ exact Hframes | ]. split.
  { intros f Hin. pose proof (Hmarks f Hin) as Hnone.
    apply lookup_app_none in Hnone as [Hl Hv].
    apply lookup_app_none. split; [ exact Hl | ].
    apply lookup_app_none. split; [ | exact Hv ].
    apply Habove. exact (proj1 (Hframes f Hin)). }
  intros s allocated Hin.
  apply lookup_app_none. split; [ | exact (Hlocals s allocated Hin) ].
  apply Habove. unfold live_locals in Hin.
  apply in_flat_map in Hin as (f & Hf & Hin).
  destruct (Hframes f Hf) as (_ & _ & Hfl). exact (proj1 (Hfl s allocated Hin)).
Qed.

(* Symbols are drawn and bound: in the newest frame, or with no frame
   in the static environment. *)
Lemma store_fresh_local (base next next' : symbol)
    (values : list (symbol * statically_allocated_value))
    (frames : list frame)
    (bindings : list (symbol * statically_allocated_value)) :
  store_fresh base next values frames ->
  (next <= next')%positive ->
  (∀ (s : symbol) (allocated : statically_allocated_value),
    In (s, allocated) bindings ->
    (next <= s)%positive
    ∧ (s < next')%positive
    ∧ allocated_below next' allocated = true) ->
  match frames with
  | [] => store_fresh base next' (bindings ++ values) []
  | Frame m kind locals :: below =>
    store_fresh base next' values (Frame m kind (bindings ++ locals) :: below)
  end.
Proof.
  intros Hstore Hnext Hbindings.
  pose proof Hstore as (Hle & Hvalues & Hframes & Hmarks & Hlocals).
  assert (Hbelow :
    ∀ s : symbol, (s < next)%positive -> lookup s bindings = None).
  { intros s Hs. apply lookup_not_in. intros allocated Hin.
    destruct (Hbindings s allocated Hin) as (Hfrom & _). lia. }
  destruct frames as [ | [m kind locals] below].
  - split; [ lia | ]. split.
    { intros s allocated Hin. apply in_app_or in Hin as [Hin | Hin].
      - destruct (Hbindings s allocated Hin) as (_ & Hs & Ha).
        exact (conj Hs Ha).
      - destruct (Hvalues s allocated Hin) as [Hs Ha].
        split; [ lia | exact (allocated_below_mono _ _ _ Hnext Ha) ]. }
    split; [ intros f [] | ].
    split; [ intros f [] | intros s allocated [] ].
  - assert (Hold : frame_drawn base next (Frame m kind locals))
      by (apply Hframes; left; reflexivity).
    split; [ lia | ]. split.
    { intros s allocated Hin. destruct (Hvalues s allocated Hin) as [Hs Ha].
      split; [ lia | exact (allocated_below_mono _ _ _ Hnext Ha) ]. }
    split.
    { intros f [<- | Hin].
      - destruct Hold as (Hm1 & Hm2 & Hl). simpl in Hm1, Hm2, Hl.
        split; [ exact Hm1 | ]. split; [ simpl; lia | ].
        simpl. intros s allocated Hin. apply in_app_or in Hin as [Hin | Hin].
        + destruct (Hbindings s allocated Hin) as (H1 & H2 & Ha).
          split; [ lia | ]. exact (conj H2 Ha).
        + destruct (Hl s allocated Hin) as (H1 & H2 & Ha).
          split; [ exact H1 | ].
          split; [ lia | exact (allocated_below_mono _ _ _ Hnext Ha) ].
      - apply (frame_drawn_mono base next next' f Hnext). apply Hframes.
        right. exact Hin. }
    split.
    { assert (Hmark :
        ∀ m' : symbol,
          (m' < next)%positive ->
          lookup m' ((locals ++ live_locals below) ++ values) = None ->
          lookup m' (((bindings ++ locals) ++ live_locals below) ++ values)
          = None).
      { intros m' Hm' Hnone. rewrite <- !app_assoc.
        apply lookup_app_none. split; [ exact (Hbelow m' Hm') | ].
        rewrite <- app_assoc in Hnone. exact Hnone. }
      intros f [<- | Hin].
      - change
          (lookup m (((bindings ++ locals) ++ live_locals below) ++ values)
           = None).
        apply Hmark; [ exact (proj1 (proj2 Hold)) | ].
        exact (Hmarks (Frame m kind locals) (or_introl eq_refl)).
      - change
          (lookup (frame_mark f)
             (((bindings ++ locals) ++ live_locals below) ++ values)
           = None).
        apply Hmark; [ exact (proj1 (proj2 (Hframes f (or_intror Hin)))) | ].
        exact (Hmarks f (or_intror Hin)). }
    intros s allocated Hin.
    change (live_locals (Frame m kind (bindings ++ locals) :: below))
      with ((bindings ++ locals) ++ live_locals below) in Hin.
    rewrite <- app_assoc in Hin. apply in_app_or in Hin as [Hin | Hin].
    + destruct (Hbindings s allocated Hin) as (Hfrom & _).
      apply lookup_not_in. intros allocated' Hin'.
      destruct (Hvalues s allocated' Hin') as [Hlt _]. lia.
    + exact (Hlocals s allocated Hin).
Qed.

(* ======================================================================== *)
(* What is held after a step                                                *)
(* ======================================================================== *)

Lemma held_below_mono (next next' : symbol) (s : state) :
  (next <= next')%positive -> held_below next s -> held_below next' s.
Proof.
  intros Hle [Hvars Hconts]. split.
  - unfold env_values_all in *. revert Hvars. apply Forall_impl.
    intros [x v]. exact (value_below_mono next next' v Hle).
  - revert Hconts. apply conts_values_all_impl. intros v.
    exact (value_below_mono next next' v Hle).
Qed.

Lemma value_below_parts_in (next : symbol) (v w : value) :
  value_below next v = true ->
  In w (value_parts v) ->
  value_below next w = true.
Proof. intros Hv. exact (forallb_in _ _ w (value_below_parts next v Hv)). Qed.

Lemma value_below_move (next : symbol) (slot slot' : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  value_below next (Value_closure_of_set slot function_decls value_slots)
  = true ->
  value_below next (Value_closure_of_set slot' function_decls value_slots)
  = true.
Proof. exact (fun H => H). Qed.

Lemma value_below_partial (next : symbol) (callee : value)
    (args : list value) (args_arity : arity) :
  value_below next (Value_partial_application callee args args_arity) = true
  <-> value_below next callee = true
      ∧ Forall (fun v => value_below next v = true) args.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

(* What a symbol is bound to names no symbol at or above the counter, if
   that is so of everything bound. *)
Lemma value_below_static (next : symbol) (statics : static_env) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    allocated_below next allocated = true) ->
  ∀ (s : symbol) (allocated : statically_allocated_value),
    value_below next (Value_symbol s) = true ->
    lookup s (static_values statics) = Some allocated ->
    (∀ w : value,
      In w (allocated_values allocated) -> value_below next w = true)
    ∧ (∀ s' : symbol,
        In s' (allocated_symbols allocated) ->
        value_below next (Value_symbol s') = true).
Proof.
  intros Hbound s allocated _ Hlookup.
  pose proof (Hbound s allocated Hlookup) as Ha.
  rewrite allocated_below_parts in Ha.
  apply andb_true_iff in Ha as [Hvalues Hsymbols]. split.
  - exact (fun w => forallb_in _ _ w Hvalues).
  - exact (fun s' => forallb_in _ _ s' Hsymbols).
Qed.

(* ======================================================================== *)
(* The cells of the heap                                                    *)
(* ======================================================================== *)

(* Every value that the heap holds names no symbol at or above [next]:
   the fourth part of FlambdaRegionsInvariantSpec.cells_live, as
   FlambdaHeapSpec.heap_values_all. *)
Definition cells_below (next : symbol) (hp : heap) : Prop :=
  heap_values_all (fun v => value_below next v = true) hp.

Lemma cells_below_forallb (next : symbol) (hp : heap) :
  forallb (fun '(_, c) => forallb (value_below next) (values_of_cell c)) hp
  = true
  <-> cells_below next hp.
Proof.
  unfold cells_below, heap_values_all. rewrite forallb_forall. split.
  - intros H l c v Hin Hv. specialize (H (l, c) Hin). cbv beta iota in H.
    exact (forallb_in _ _ v H Hv).
  - intros H [l c] Hin. apply forallb_forall. intros v Hv.
    exact (H l c v Hin Hv).
Qed.

Lemma cells_below_mono (next next' : symbol) (hp : heap) :
  (next <= next')%positive -> cells_below next hp -> cells_below next' hp.
Proof.
  intros Hle H l c v Hin Hv.
  exact (value_below_mono _ _ v Hle (H l c v Hin Hv)).
Qed.

(* What a [Let] binds, and what it leaves in the heap, name no symbol at
   or above [next], if what the static environment binds, the values of
   its simples and the cells of the heap name none. *)
Lemma exec_named_below (next : symbol) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  (∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    allocated_below next allocated = true) ->
  (∀ (sim : simple) (w : value),
    In sim (simples_of_named d) ->
    eval_simple vars sim = Some w ->
    value_below next w = true) ->
  cells_below next hp ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  value_below next v = true ∧ cells_below next hp'.
Proof.
  intros Hbound Hsimples Hcells Hexec.
  assert (Hcases :
    (∃ (p : prim) (args : list simple), d = Prim p args)
    ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d;
      first [ left; eexists _, _; reflexivity | right; intros; discriminate ]. }
  destruct Hcases as [(p & args & ->) | Hnot].
  - apply
      (exec_named_preserves (fun v => value_below next v = true) statics hp
         hp' rd rd' vars p args v);
      try assumption.
    + intros c. split; reflexivity.
    + intros tag fields. simpl. apply forallb_forall.
    + intros elements. simpl. apply forallb_forall.
    + intros s tag fields Hs Hlookup.
      exact (proj1 (value_below_static next statics Hbound s _ Hs Hlookup)).
    + intros s elements Hs Hlookup.
      exact (proj1 (value_below_static next statics Hbound s _ Hs Hlookup)).
    + intros l. reflexivity.
    + intros vs Hvs arg Harg.
      exact
        (proj1 (Forall_forall _ vs)
           (eval_simples_forall _ vars args vs Hsimples Hvs) arg Harg).
  - rewrite (exec_named_not_prim statics hp rd vars d Hnot) in Hexec.
    destruct (eval_named statics vars d) as [v0 | ] eqn:Hnamed;
      [ | discriminate ].
    simpl in Hexec. injection Hexec as -> <- _.
    split; [ | exact Hcells ].
    destruct (is_allocation d) eqn:His.
    + destruct (eval_named_makes _ _ d v His Hnamed) as [Hmade Hparts].
      apply (value_below_made next v Hmade). apply forallb_forall.
      intros w Hin. destruct (Hparts w Hin) as (sim & Hsim & Hw).
      exact (Hsimples sim w Hsim Hw).
    + exact
        (eval_named_reads (fun v => value_below next v = true) statics
           (fun c _ => eq_refl) (value_below_parts_in next)
           (value_below_move next) (value_below_static next _ Hbound) vars d
           v His Hsimples Hnamed).
Qed.

(* What the static environment of a step binds names no symbol at or
   above a counter that the step does not pass. *)
Lemma live_statics_below (rs : region_state) (next' : symbol) :
  symbols_fresh rs = true ->
  (rs_next rs <= next')%positive ->
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values (live_statics rs)) = Some allocated ->
    allocated_below next' allocated = true.
Proof.
  intros Hfresh Hnext s allocated Hlookup.
  pose proof (proj1 (symbols_fresh_parts rs) Hfresh) as (_ & _ & Hstore).
  destruct
    (store_bound_below _ _ _ _ s allocated Hstore (lookup_in _ _ _ Hlookup))
    as [_ Ha].
  exact (allocated_below_mono _ _ _ Hnext Ha).
Qed.

(* The values of the simples of a [Let] that the state is at. *)
Lemma let_simples_below (rs : region_state) (next' : symbol) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  symbols_fresh rs = true ->
  (rs_next rs <= next')%positive ->
  state_expr (rs_state rs) = Let x kind d body ->
  ∀ (sim : simple) (w : value),
    In sim (simples_of_named d) ->
    eval_simple (state_vars (rs_state rs)) sim = Some w ->
    value_below next' w = true.
Proof.
  intros Hfresh Hnext Hexpr sim w.
  pose proof (proj1 (symbols_fresh_parts rs) Hfresh)
    as ((He & _ & _) & Hheld & Hstore).
  pose proof Hstore as (Hle & _).
  assert (Hle' : (rs_base rs <= next')%positive) by lia.
  pose proof (held_below_mono _ _ _ Hnext Hheld) as Hheld'.
  rewrite Hexpr in He.
  exact
    (named_simples_below _ _ _ x kind d body sim w Hle' He (proj1 Hheld')).
Qed.

(* The values of a state after a step, and those of an exit, name no
   symbol at or above a counter that the step does not pass. *)
Lemma region_step_below (rs : region_state) (next' : symbol) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  (rs_next rs <= next')%positive ->
  region_step_vals_ok (fun v => value_below next' v = true) rs
    (region_step rs).
Proof.
  intros Hfresh Hcells Hnext.
  pose proof (proj1 (symbols_fresh_parts rs) Hfresh)
    as ((He & _ & _) & Hheld & Hstore).
  pose proof Hstore as (Hle & _).
  assert (Hle' : (rs_base rs <= next')%positive) by lia.
  pose proof (held_below_mono _ _ _ Hnext Hheld) as Hheld'.
  pose proof (live_statics_below rs next' Hfresh Hnext) as Hbound.
  apply (region_step_values _ (value_below_partial next') rs).
  - apply reads_fields;
      [ exact (value_below_parts_in next')
      | exact (value_below_static next' _ Hbound) ].
  - exact Hheld'.
  - intros x kind d body v hp' rd' Hexpr _ _ _ Hexec.
    exact
      (proj1
         (exec_named_below next' _ _ _ _ _ _ d v Hbound
            (let_simples_below rs next' x kind d body Hfresh Hnext Hexpr)
            (cells_below_mono _ _ _ Hnext Hcells) Hexec)).
  - intros sim v Hin Hv.
    exact
      (eval_simple_below _ _ _ sim v Hle'
         (evaluated_simples_below _ _ sim He Hin) (proj1 Hheld') Hv).
  - reflexivity.
  - intros l. reflexivity.
Qed.

Lemma region_step_held (rs rs' : region_state) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  region_step rs = Region_step rs' ->
  held_below (rs_next rs') (rs_state rs').
Proof.
  intros Hfresh Hcells Hstep.
  destruct (region_step_draws_upwards rs rs' Hstep) as [_ Hnext].
  apply Pos.leb_le in Hnext.
  pose proof (region_step_below rs (rs_next rs') Hfresh Hcells Hnext)
    as Hvalues.
  rewrite Hstep in Hvalues. apply Hvalues.
  intros s _ Hlt. apply Pos.ltb_lt. exact Hlt.
Qed.

(* The cells of the heap after a step. *)
Lemma region_step_cells_below (rs rs' : region_state) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  region_step rs = Region_step rs' ->
  cells_below (rs_next rs') (state_heap (rs_state rs')).
Proof.
  intros Hfresh Hcells Hstep.
  destruct (region_step_draws_upwards rs rs' Hstep) as [_ Hnext].
  apply Pos.leb_le in Hnext.
  destruct (region_step_heap rs rs' Hstep)
    as [-> | (x & kind & d & body & v & rd' & Hexpr & _ & Hexec)].
  - exact (cells_below_mono _ _ _ Hnext Hcells).
  - exact
      (proj2
         (exec_named_below (rs_next rs') _ _ _ _ _ _ d v
            (live_statics_below rs _ Hfresh Hnext)
            (let_simples_below rs _ x kind d body Hfresh Hnext Hexpr)
            (cells_below_mono _ _ _ Hnext Hcells) Hexec)).
Qed.

(* ======================================================================== *)
(* What is bound after a step                                               *)
(* ======================================================================== *)

Lemma region_step_store (rs rs' : region_state) :
  symbols_fresh rs = true ->
  region_step rs = Region_step rs' ->
  store_fresh (rs_base rs') (rs_next rs')
    (static_values (state_statics (rs_state rs'))) (rs_frames rs').
Proof.
  intros Hfresh.
  pose proof (proj1 (symbols_fresh_parts rs) Hfresh)
    as ((He & _ & _) & [Hvars _] & Hstore).
  pose proof Hstore as (Hle & _).
  apply
    (region_step_cases rs
       (fun result =>
          result = Region_step rs' ->
          store_fresh (rs_base rs') (rs_next rs')
            (static_values (state_statics (rs_state rs'))) (rs_frames rs'))).
  - discriminate.
  - intros x kind d mark_kind body _ Hstep.
    rewrite (begin_region_step _ _ _ _ _ _ Hstep).
    exact (store_fresh_begin _ _ _ _ mark_kind Hstore).
  - intros x kind d ended_by region body _ Hstep.
    destruct (end_region_step _ _ _ _ _ _ _ Hstep)
      as (m & above & f & below & counts & _ & Hframes & ->).
    change (rs_frames (count_uses rs)) with (rs_frames rs) in Hframes.
    rewrite Hframes in Hstore.
    exact (store_fresh_end _ _ _ above f below Hstore).
  - intros x kind d body y region body' Hexpr _ _ Hcell Hstep.
    destruct (local_allocation_step _ _ _ _ _ _ _ _ Hcell Hstep)
      as (v & sym & bindings & His & Hnamed & Hlocalise & _ & _ & _ & Hbase
          & Hnext & _ & Hframes).
    destruct (localise_makes _ _ _ _ Hlocalise) as (_ & _ & Hbound).
    destruct (eval_named_makes _ _ d v His Hnamed) as [_ Hparts].
    cbn [count_uses with_counts rs_state rs_frames rs_base rs_next]
      in Hbase, Hnext, Hframes, Hbound, Hparts.
    rewrite Hexpr in He.
    pose proof (advance_le (rs_next rs) (length bindings)) as Hadvance.
    rewrite <- Hnext in Hadvance, Hbound.
    assert (Hbindings :
      ∀ (s : symbol) (allocated : statically_allocated_value),
        In (s, allocated) bindings ->
        (rs_next rs <= s)%positive
        ∧ (s < rs_next rs')%positive
        ∧ allocated_below (rs_next rs') allocated = true).
    { intros s allocated Hin.
      destruct (Hbound s allocated Hin)
        as (Hfrom & Hbelow & Hvalues & Hsymbols).
      split; [ exact Hfrom | ]. split; [ exact Hbelow | ].
      rewrite allocated_below_parts. apply andb_true_iff. split.
      - rewrite Hvalues. apply forallb_forall. intros w Hw.
        destruct (Hparts w Hw) as (sim & Hsim & Heval).
        apply (value_below_mono (rs_next rs)); [ exact Hadvance | ].
        exact
          (named_simples_below _ _ _ x kind d body sim w Hle He Hvars Hsim
             Heval).
      - apply forallb_forall. intros s' Hs'.
        destruct (Hsymbols s' Hs') as (allocated' & Hin').
        apply Pos.ltb_lt. exact (proj1 (proj2 (Hbound s' allocated' Hin'))). }
    pose proof
      (store_fresh_local _ _ (rs_next rs') _ _ bindings Hstore Hadvance
         Hbindings)
      as Hlocal.
    rewrite Hbase. revert Hlocal Hframes.
    destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
      intros Hlocal [Hf Hs]; rewrite Hf, Hs; exact Hlocal.
  - intros x kind d body y region body' _ _ _ _ Hstep.
    destruct (local_cell_allocation_step _ _ _ _ _ _ _ Hstep)
      as (l & hp & rd & _ & _ & Hstate & Hf & Hb & Hn & _).
    rewrite Hstate, Hf, Hb, Hn. exact Hstore.
  - intros x kind d body _ _ _ _ Hstep.
    destruct (plain_step_step _ _ Hstep) as (s' & _ & ->). exact Hstore.
  - intros bindings body Hexpr _ _ Hstep.
    destruct (static_step_step (count_uses rs) rs' bindings body Hexpr Hstep)
      as (statics' & Heval & ->).
    destruct (eval_static_bindings_parts _ _ _ _ Heval)
      as (new_code & new_values & _ & Hvalues & _ & Hnew).
    cbn [count_uses with_counts rs_state rs_frames rs_base rs_next
         state_statics]
      in Hvalues |- *.
    rewrite Hvalues. apply store_fresh_static; [ exact Hstore | ].
    rewrite Hexpr in He. simpl in He. apply andb_true_iff in He as [Hb _].
    intros s allocated Hin.
    destruct (Hnew s allocated Hin) as (const & Hbinding & Hconst).
    pose proof (forallb_in _ _ _ Hb Hbinding) as Hbelow. simpl in Hbelow.
    apply andb_true_iff in Hbelow as [Hs Hc].
    split; [ apply Pos.ltb_lt; exact Hs | ].
    exact (eval_static_const_below _ _ _ const allocated Hle Hc Hvars Hconst).
  - intros ac _ Hstep.
    destruct (region_apply_cont_step _ _ ac Hstep) as (Hf & Hb & Hn & Hs).
    rewrite Hf, Hb, Hn, Hs. exact Hstore.
  - intros _ _ Hstep.
    destruct (plain_step_step _ _ Hstep) as (s' & _ & ->). exact Hstore.
Qed.

(* ======================================================================== *)
(* The first part of the invariant                                          *)
(* ======================================================================== *)

(* The state that a run starts in has it. *)
Theorem symbols_fresh_initial (e : expr) (statics : static_env)
    (base : symbol) :
  region_names_fresh base e statics = true ->
  symbols_fresh (initial_region_state e statics base) = true.
Proof.
  intros H. unfold region_names_fresh, statics_below in H.
  apply andb_true_iff in H as [H _].
  apply andb_true_iff in H as [He H].
  apply andb_true_iff in H as [Hcode Hvalues].
  apply symbols_fresh_parts. split; [ exact (conj He (conj I Hcode)) | ].
  split; [ exact (conj (Forall_nil _) I) | ].
  split; [ apply Pos.le_refl | ]. split.
  { intros s allocated Hin.
    pose proof (forallb_in _ _ (s, allocated) Hvalues Hin) as Hs.
    cbv beta iota in Hs. apply andb_true_iff in Hs as [Hs Ha].
    split; [ apply Pos.ltb_lt; exact Hs | exact Ha ]. }
  split; [ intros f [] | ].
  split; [ intros f [] | intros s allocated [] ].
Qed.

(* A step keeps it. *)
Theorem region_step_keeps_symbols_fresh (rs rs' : region_state) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  region_step rs = Region_step rs' ->
  symbols_fresh rs' = true.
Proof.
  intros Hfresh Hcells Hstep. apply symbols_fresh_parts. split; [ | split ].
  - apply (region_step_keeps_written_below rs rs'); [ | exact Hstep ].
    exact (proj1 (proj1 (symbols_fresh_parts rs) Hfresh)).
  - exact (region_step_held rs rs' Hfresh Hcells Hstep).
  - exact (region_step_store rs rs' Hfresh Hstep).
Qed.

(* The state that a step exits with has it, and the values of the exit
   name no symbol at or above its counter. *)
Theorem region_step_exit_symbols_fresh (rs : region_state) (k : continuation)
    (vs : list value) (rs' : region_state) :
  symbols_fresh rs = true ->
  region_step rs = Region_exit k vs rs' ->
  symbols_fresh rs' = true.
Proof.
  intros Hfresh Hexit. rewrite (region_step_exit rs rs' k vs Hexit).
  exact Hfresh.
Qed.

Lemma region_step_exit_values_below (rs : region_state) (k : continuation)
    (vs : list value) (rs' : region_state) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  region_step rs = Region_exit k vs rs' ->
  forallb (value_below (rs_next rs')) vs = true.
Proof.
  intros Hfresh Hcells Hexit.
  pose proof
    (region_step_below rs (rs_next rs) Hfresh Hcells (Pos.le_refl _))
    as Hvalues.
  rewrite Hexit in Hvalues. rewrite (region_step_exit rs rs' k vs Hexit).
  exact (proj2 (forallb_Forall _ vs) Hvalues).
Qed.

(* What the expression of a state names, after a step. *)
Lemma region_step_keeps_expr_below (rs rs' : region_state) :
  symbols_fresh rs = true ->
  region_step rs = Region_step rs' ->
  expr_below (rs_base rs') (state_expr (rs_state rs')) = true.
Proof.
  intros Hfresh Hstep.
  exact
    (proj1
       (region_step_keeps_written_below rs rs'
          (proj1 (proj1 (symbols_fresh_parts rs) Hfresh)) Hstep)).
Qed.

(* ======================================================================== *)
(* What is gone stays gone                                                  *)
(* ======================================================================== *)

(* FlambdaRegionsProofs.gone_stays_gone, with the premise that a run
   keeps. Of FlambdaRegionsSpec.region_names_fresh that proof uses what
   the expression of the state names, and nothing else. *)
Theorem gone_stays_gone_fresh (rs rs' : region_state) (s : symbol) :
  symbols_fresh rs = true ->
  region_step rs = Region_step rs' ->
  gone rs s ->
  gone rs' s.
Proof.
  intros Hfresh Hstep (Hdrawn & Hnext & Hstack & Hlookup).
  pose proof (proj1 (symbols_fresh_parts rs) Hfresh)
    as ((Hexpr & _ & _) & _ & _).
  destruct (region_step_shape _ _ Hstep) as [Hbase Hshape].
  unfold gone, drawn. rewrite Hbase. split; [ exact Hdrawn | ].
  apply Pos.ltb_lt in Hnext.
  unfold drawn in Hdrawn. apply Pos.leb_le in Hdrawn.
  unfold live_statics in Hlookup |- *. simpl in Hlookup |- *.
  apply lookup_app_none in Hlookup as [Hlocals Hstatics].
  destruct Hshape
    as [Hframes Hnext' Hstatics' _
       | bindings body He Hframes Hnext' Hstatics'
       | kind Hframes Hnext' Hstatics' _
       | above f Hframes Hnext' Hstatics' _
       | v sym bindings Hlocalise Hnext' Hframes _].
  - rewrite Hframes, Hnext', Hstatics'.
    split; [ apply Pos.ltb_lt; exact Hnext | ].
    split; [ exact Hstack | ].
    apply lookup_app_none. split; assumption.
  - rewrite Hframes, Hnext'.
    split; [ apply Pos.ltb_lt; exact Hnext | ].
    split; [ exact Hstack | ].
    apply lookup_app_none. split; [ exact Hlocals | ].
    rewrite He in Hexpr. simpl in Hexpr.
    apply andb_true_iff in Hexpr as [Hbindings _].
    rewrite
      (eval_static_bindings_lookup _ _ _ _ _ s Hbindings Hdrawn Hstatics').
    exact Hstatics.
  - rewrite Hframes, Hnext', Hstatics'.
    split; [ apply Pos.ltb_lt; lia | ].
    split.
    + simpl. rewrite Hstack, orb_false_r. apply Pos.eqb_neq. lia.
    + simpl. apply lookup_app_none. split; assumption.
  - rewrite Hnext', Hstatics'. rewrite Hframes in Hstack, Hlocals.
    split; [ apply Pos.ltb_lt; exact Hnext | ].
    rewrite on_stack_app in Hstack.
    apply orb_false_iff in Hstack as [_ Hstack]. simpl in Hstack.
    apply orb_false_iff in Hstack as [_ Hstack].
    split; [ exact Hstack | ].
    rewrite live_locals_app in Hlocals.
    apply lookup_app_none in Hlocals as [_ Hlocals]. simpl in Hlocals.
    apply lookup_app_none in Hlocals as [_ Hlocals].
    apply lookup_app_none. split; assumption.
  - rewrite Hnext'.
    pose proof (advance_le (rs_next rs) (length bindings)) as Hadvance.
    split; [ apply Pos.ltb_lt; lia | ].
    pose proof (localise_lookup_below _ _ _ _ s Hlocalise Hnext)
      as Hbindings.
    destruct (rs_frames rs) as [ | f frames].
    + destruct Hframes as [-> ->]. split; [ reflexivity | ].
      simpl. apply lookup_app_none. split; assumption.
    + destruct Hframes as [-> ->]. split; [ exact Hstack | ].
      simpl in Hlocals |- *.
      apply lookup_app_none in Hlocals as [Hf Hlocals].
      apply lookup_app_none. split; [ | exact Hstatics ].
      rewrite <- app_assoc.
      apply lookup_app_none. split; [ exact Hbindings | ].
      apply lookup_app_none. split; assumption.
Qed.
