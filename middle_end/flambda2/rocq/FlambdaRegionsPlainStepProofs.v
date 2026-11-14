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

(* The erasure of regions (FlambdaRegionsErasureSpec.v), at the
   expressions where the machine of regions does what [step] does: a
   [Let] that is neither of a primitive of regions nor followed by a
   [Local_allocation], a [Let_static], the two [Let_cont], an [Apply],
   an [Apply_cont] that gives no region, a [Switch] and [Invalid].
   Nothing here needs review: no statement of a Spec file is proved
   here. The rules of regions, and the two statements of
   FlambdaRegionsErasureSpec.v, are not here.

   The state of the machine of regions and the state that runs the
   erased term are related by
   FlambdaRegionsErasedProofs.state_erased. Where the first has a step
   of [plain_step] or of [static_step], the second has one step of
   [step], to a state related again; where the first exits, the second
   exits through the same continuation with the values related where
   they may be evaluated or are locations ([plain_step_erased],
   [static_step_erased],
   and [region_step_shared] for a step of [region_step] at such an
   expression).

   What is asked of the first state beside the relation, and comes
   from the invariant of the machine
   (FlambdaRegionsInvariantSpec.v): the expression has no symbol that
   the machine drew ([expr_below]), and at an [Apply_cont] no value of
   an argument is of the kind [Region]. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaClosuresSpec
  FlambdaClosuresProofs FlambdaLiftProofs FlambdaRegionsSpec
  FlambdaRegionsProofs FlambdaRegionsErasureSpec FlambdaRegionsErasedProofs.
Import ListNotations.

#[local] Opaque live_statics.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

Lemma eval_static_const_erased (rs : region_state) (vars vars' : env)
    (const : static_const) (allocated : statically_allocated_value) :
  env_erased rs vars vars' ->
  static_const_below (rs_base rs) const = true ->
  simples_usable rs vars (simples_of_static_const const) ->
  eval_static_const vars const = Some allocated ->
  ∃ allocated' : statically_allocated_value,
    eval_static_const vars' const = Some allocated'
    ∧ allocated_erased rs allocated allocated'.
Proof.
  intros Hvars Hbelow Husable. unfold static_const_below in Hbelow.
  apply andb_true_iff in Hbelow as [Hbelow Hsyms].
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl in Hbelow, Husable |- *.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct
      (eval_value_slots_erased rs vars vars' slots Hvars Hbelow Husable vs
         Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_static_closure, Hvs.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct
      (eval_value_slots_erased rs vars vars' slots Hvars Hbelow Husable vs
         Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_static_closure_of_set; [ | exact Hvs ].
    intros slot s Hin. rewrite forallb_forall in Hsyms.
    exact (drawn_below rs s (Hsyms (slot, s) Hin)).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    destruct
      (eval_simples_erased rs vars vars' fields Hvars Hbelow Husable vs
         Hfields)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_static_block, Hvs.
  - apply andb_true_iff in Hbelow as [Hbelow _].
    destruct (eval_simple vars contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    destruct
      (eval_simple_erased rs vars vars' contents v Hvars Hbelow Hcontents
         (Husable contents v (or_introl eq_refl) Hcontents))
      as (v' & -> & Hv).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_static_boxed_number, Hv.
  - intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_static_string.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    destruct
      (eval_simples_erased rs vars vars' elements Hvars Hbelow Husable vs
         Helements)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Erased_static_immutable_array, Hvs.
Qed.

(* A state of the machine of regions with another static environment:
   the state after a group of static bindings, but for its
   expression. *)
Definition with_region_statics (rs : region_state) (statics : static_env)
    : region_state :=
  Region_state (with_statics (rs_state rs) statics) (rs_frames rs)
    (rs_base rs) (rs_next rs) (rs_counts rs) (rs_cells rs).

Lemma with_region_statics_same (rs : region_state) :
  with_region_statics rs (state_statics (rs_state rs)) = rs.
Proof.
  unfold with_region_statics. rewrite with_statics_same. destruct rs.
  reflexivity.
Qed.

(* A symbol that is not drawn, bound in the static environment of the
   state where it was not bound: what was bound stays bound to the same
   thing, and no drawn symbol becomes bound. *)
Lemma region_extends_symbol (rs : region_state) (s : symbol)
    (allocated : statically_allocated_value) :
  drawn rs s = false ->
  lookup s (static_values (state_statics (rs_state rs))) = None ->
  region_extends rs
    (with_region_statics rs
       (Static_env (static_code (state_statics (rs_state rs)))
          ((s, allocated) :: static_values (state_statics (rs_state rs))))).
Proof.
  intros Hnot_drawn Hfresh. split; [ reflexivity | ].
  rewrite !live_statics_values. simpl. split; [ | split ].
  - intros s0 allocated0. rewrite !lookup_app.
    destruct (lookup s0 (live_locals (rs_frames rs)));
      [ exact (fun H => H) | ].
    simpl. destruct (Pos.eqb_spec s0 s) as [-> | _]; [ | exact (fun H => H) ].
    rewrite Hfresh. discriminate.
  - intros s0 Hdrawn. rewrite !lookup_app.
    destruct (lookup s0 (live_locals (rs_frames rs))); [ discriminate | ].
    simpl. destruct (Pos.eqb_spec s0 s) as [-> | _]; [ | exact (fun H => H) ].
    rewrite Hnot_drawn in Hdrawn. discriminate.
  - intros l. reflexivity.
Qed.

(* Code added to the static environment of the state: the symbols are
   as they were. *)
Lemma region_extends_code (rs : region_state) (code : list (code_id * code)) :
  region_extends rs
    (with_region_statics rs
       (Static_env code (static_values (state_statics (rs_state rs))))).
Proof.
  apply region_extends_same; [ reflexivity | | intros l; reflexivity ].
  rewrite !live_statics_values. reflexivity.
Qed.

(* A group of static bindings that the machine of regions adds to the
   static environment of its state, the group erased is added to the
   second static environment: the code erased, the symbols bound to
   the same things with their parts erased. *)
Lemma eval_static_bindings_erased (vars vars' : env)
    (bindings : list static_binding) :
  ∀ (rs : region_state) (statics' statics1 : static_env),
    env_erased rs vars vars' ->
    statics_erased rs statics' ->
    forallb (binding_below (rs_base rs)) bindings = true ->
    forallb binding_returns_no_region bindings = true ->
    simples_usable rs vars (flat_map simples_of_binding bindings) ->
    eval_static_bindings vars bindings (state_statics (rs_state rs))
    = Some statics1 ->
    ∃ statics1' : static_env,
      eval_static_bindings vars' (map erase_binding bindings) statics'
      = Some statics1'
      ∧ statics_erased (with_region_statics rs statics1) statics1'
      ∧ region_extends rs (with_region_statics rs statics1).
Proof.
  induction bindings as [ | binding bindings IH];
    intros rs statics' statics1 Hvars Hstatics Hbelow Hreturns Husable.
  - simpl. intros [= <-]. exists statics'. split; [ reflexivity | ].
    rewrite with_region_statics_same.
    split; [ exact Hstatics | apply region_extends_refl ].
  - simpl in Hbelow, Hreturns.
    apply andb_true_iff in Hbelow as [Hbinding Hbelow].
    apply andb_true_iff in Hreturns as [Hreturn Hreturns].
    destruct Hstatics as (Hcode & Hcode_returns & Hsymbols & Hbound).
    destruct binding as [id c | id | s const]; simpl.
    + (* Code. *)
      rewrite Hcode.
      destruct (lookup id (static_code (state_statics (rs_state rs))))
        eqn:Hid;
        [ discriminate | ].
      simpl. intros Heval.
      pose proof
        (region_extends_code rs
           ((id, c) :: static_code (state_statics (rs_state rs))))
        as Hextends.
      destruct
        (IH
           (with_region_statics rs
              (Static_env
                 ((id, c) :: static_code (state_statics (rs_state rs)))
                 (static_values (state_statics (rs_state rs)))))
           (Static_env ((id, erase_code c) :: static_code statics')
              (static_values statics'))
           statics1)
        as (statics1' & Heval' & Hstatics1 & Hextends1).
      * exact (env_erased_extend _ _ _ _ Hextends Hvars).
      * split; [ | split; [ | split ] ].
        -- intros id0. simpl.
           destruct (Pos.eqb id0 id); [ reflexivity | apply Hcode ].
        -- intros id0 c0. simpl.
           destruct (Pos.eqb id0 id);
             [ intros [= <-]; exact Hreturn | apply Hcode_returns ].
        -- intros s0 allocated Hnot_drawn Hs0.
           change (drawn rs s0 = false) in Hnot_drawn.
           rewrite live_statics_values in Hs0. simpl in Hs0.
           rewrite <- live_statics_values in Hs0.
           destruct (Hsymbols s0 allocated Hnot_drawn Hs0)
             as (allocated' & Hs0' & Hallocated).
           exists allocated'. split; [ exact Hs0' | ].
           exact (allocated_erased_extend _ _ _ _ Hextends Hallocated).
        -- exact Hbound.
      * exact Hbelow.
      * exact Hreturns.
      * intros s0 v Hin Hs0.
        rewrite (value_usable_extends _ _ v Hextends).
        exact (Husable s0 v Hin Hs0).
      * exact Heval.
      * exists statics1'. split; [ exact Heval' | ].
        split; [ exact Hstatics1 | ].
        exact (region_extends_trans _ _ _ Hextends Hextends1).
    + (* Deleted code. *)
      intros Heval.
      exact (IH rs statics' statics1 Hvars
               (conj Hcode (conj Hcode_returns (conj Hsymbols Hbound)))
               Hbelow Hreturns Husable Heval).
    + (* A symbol. *)
      simpl in Hbinding. apply andb_true_iff in Hbinding as [Hs Hconst].
      pose proof (drawn_below rs s Hs) as Hnot_drawn.
      destruct (lookup s (static_values (state_statics (rs_state rs))))
        eqn:Hfresh;
        [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hallocated;
        [ | discriminate ].
      simpl. intros Heval.
      assert (Hfresh' : lookup s (static_values statics') = None).
      { destruct (lookup s (static_values statics')) as [a' | ] eqn:Hs';
          [ | reflexivity ].
        contradiction (Hbound s a' Hs' Hfresh). }
      rewrite Hfresh'.
      destruct
        (eval_static_const_erased rs vars vars' const allocated Hvars Hconst
           (fun s0 v Hin => Husable s0 v (in_or_app _ _ _ (or_introl Hin)))
           Hallocated)
        as (allocated' & -> & Hrelated).
      simpl.
      pose proof (region_extends_symbol rs s allocated Hnot_drawn Hfresh)
        as Hextends.
      destruct
        (IH
           (with_region_statics rs
              (Static_env (static_code (state_statics (rs_state rs)))
                 ((s, allocated)
                  :: static_values (state_statics (rs_state rs)))))
           (Static_env (static_code statics')
              ((s, allocated') :: static_values statics'))
           statics1)
        as (statics1' & Heval' & Hstatics1 & Hextends1).
      * exact (env_erased_extend _ _ _ _ Hextends Hvars).
      * split; [ | split; [ | split ] ].
        -- exact Hcode.
        -- exact Hcode_returns.
        -- intros s0 allocated0 Hnot_drawn0 Hs0.
           change (drawn rs s0 = false) in Hnot_drawn0.
           rewrite live_statics_values, lookup_app in Hs0. simpl in Hs0.
           simpl.
           destruct (lookup s0 (live_locals (rs_frames rs)))
             as [local | ] eqn:Hlocal.
           ++ (* Bound in a frame: then it was bound before, and the
                 second static environment bound it; so it is not the
                 symbol of the binding. *)
              assert (Hbefore :
                lookup s0 (static_values (live_statics rs)) = Some allocated0).
              { rewrite live_statics_values, lookup_app, Hlocal.
                exact Hs0. }
              destruct (Hsymbols s0 allocated0 Hnot_drawn0 Hbefore)
                as (allocated0' & Hs0' & Hallocated0).
              destruct (Pos.eqb_spec s0 s) as [-> | _];
                [ rewrite Hfresh' in Hs0'; discriminate | ].
              exists allocated0'. split; [ exact Hs0' | ].
              exact (allocated_erased_extend _ _ _ _ Hextends Hallocated0).
           ++ destruct (Pos.eqb_spec s0 s) as [-> | _].
              ** injection Hs0 as <-. exists allocated'.
                 split; [ reflexivity | ].
                 exact (allocated_erased_extend _ _ _ _ Hextends Hrelated).
              ** assert (Hbefore :
                   lookup s0 (static_values (live_statics rs))
                   = Some allocated0).
                 { rewrite live_statics_values, lookup_app, Hlocal.
                   exact Hs0. }
                 destruct (Hsymbols s0 allocated0 Hnot_drawn0 Hbefore)
                   as (allocated0' & Hs0' & Hallocated0).
                 exists allocated0'. split; [ exact Hs0' | ].
                 exact (allocated_erased_extend _ _ _ _ Hextends Hallocated0).
        -- intros s0 allocated0'. simpl.
           destruct (Pos.eqb s0 s); [ discriminate | apply Hbound ].
      * exact Hbelow.
      * exact Hreturns.
      * intros s0 v Hin Hs0.
        rewrite (value_usable_extends _ _ v Hextends).
        exact (Husable s0 v (in_or_app _ _ _ (or_intror Hin)) Hs0).
      * exact Heval.
      * exists statics1'. split; [ exact Heval' | ].
        split; [ exact Hstatics1 | ].
        exact (region_extends_trans _ _ _ Hextends Hextends1).
Qed.

(* ======================================================================== *)
(* What erasing leaves as it is                                             *)
(* ======================================================================== *)

(* The term erased binds something static exactly if the term does: the
   [Let]s that are dropped bind nothing static. *)
Fixpoint erase_binds_statics (e : expr) {struct e} :
  ∀ rconts : region_conts, binds_statics (erase rconts e) = binds_statics e.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros rconts; try reflexivity.
  - assert (H :
      binds_statics (erase rconts (Let x xkind d body))
      = binds_statics (erase rconts body)).
    { destruct d as [ s | p args | | | | | ]; try reflexivity.
      destruct p; try reflexivity. simpl.
      destruct (erase_region_primitive x _ args); reflexivity. }
    rewrite H. exact (erase_binds_statics body rconts).
  - simpl. rewrite (erase_binds_statics body rconts). f_equal.
    induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
    rewrite IH. destruct binding; reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite (erase_binds_statics handler rconts),
      (erase_binds_statics body
         ((k, region_params (Handler params metadata handler)) :: rconts)).
    reflexivity.
  - simpl. rewrite (erase_binds_statics body _). f_equal.
    generalize (map (fun '(k, h) => (k, region_params h)) handlers ++ rconts).
    intros rc.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite (erase_binds_statics handler rc), IH. reflexivity.
Qed.

Lemma erase_handlers_bind_statics (rconts : region_conts)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (erase_handlers rconts handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, erase_handlers.
  induction handlers as [ | [k [params metadata handler]] handlers IH];
    simpl; [ reflexivity | ].
  rewrite (erase_binds_statics handler rconts), IH. reflexivity.
Qed.

(* Code erased is the code with its body erased. *)
Lemma erase_code_params (c : code) :
  code_params (erase_code c) = code_params c.
Proof. destruct c. reflexivity. Qed.

Lemma erase_code_my_closure (c : code) :
  code_my_closure (erase_code c) = code_my_closure c.
Proof. destruct c. reflexivity. Qed.

Lemma erase_code_return_continuation (c : code) :
  code_return_continuation (erase_code c) = code_return_continuation c.
Proof. destruct c. reflexivity. Qed.

Lemma erase_code_exn_continuation (c : code) :
  code_exn_continuation (erase_code c) = code_exn_continuation c.
Proof. destruct c. reflexivity. Qed.

Lemma erase_code_params_arity (c : code) :
  code_params_arity (erase_code c) = code_params_arity c.
Proof. destruct c. reflexivity. Qed.

Lemma erase_code_result_arity (c : code) :
  code_result_arity (erase_code c) = code_result_arity c.
Proof. destruct c. reflexivity. Qed.

Lemma erase_code_is_tupled (c : code) :
  code_is_tupled (erase_code c) = code_is_tupled c.
Proof. destruct c. reflexivity. Qed.

Lemma erase_code_body (c : code) :
  code_body (erase_code c) = erase [] (code_body c).
Proof. destruct c. reflexivity. Qed.

Lemma code_body_returns_no_region (c : code) :
  code_returns_no_region c = true ->
  calls_return_no_region [] (code_body c) = true.
Proof. destruct c. exact (fun H => H). Qed.

(* ======================================================================== *)
(* The result of a step                                                     *)
(* ======================================================================== *)

(* The state of the machine of regions after a step of [step] to [s1]
   under the static environment of the frames
   (FlambdaRegionsSpec.plain_step). *)
Definition plain_next (rs : region_state) (s1 : state) : region_state :=
  Region_state (with_statics s1 (state_statics (rs_state rs))) (rs_frames rs)
    (rs_base rs) (rs_next rs) (rs_counts rs) (rs_cells rs).

(* What the second state answers to a result of [step] on the first,
   under the static environment of the frames: nothing is asked if the
   first is undefined; an exit is answered by an exit through the same
   continuation, with the values related where they may be evaluated
   or are locations, the heaps related and the same reads; a step, by
   a step to a state related. *)
Definition step_result_erased (rs : region_state)
    (result result' : step_result) : Prop :=
  match result with
  | Undefined => True
  | Exit k vs hp rd =>
    ∃ (vs' : list value) (hp' : heap),
      result' = Exit k vs' hp' rd
      ∧ heaps_erased rs hp hp'
      ∧ Forall2 (value_erased_if_usable rs) vs vs'
  | Step s1 =>
    ∃ s1' : state, result' = Step s1' ∧ state_erased (plain_next rs s1) s1'
  end.

(* The state after such a step is related to a state whose parts are
   related to its parts, under the state before. *)
Lemma state_erased_plain_next (rs : region_state) (rconts : region_conts)
    (e1 : expr) (vars1 vars1' : env) (conts1 conts1' : cont_env)
    (statics statics' : static_env) (hp hp' : heap) (rd : read_record) :
  calls_return_no_region rconts e1 = true ->
  conts_agree rconts conts1 ->
  env_erased rs vars1 vars1' ->
  conts_erased rs conts1 conts1' ->
  statics_erased rs statics' ->
  heaps_erased rs hp hp' ->
  state_erased (plain_next rs (State e1 vars1 conts1 statics hp rd))
    (State (erase rconts e1) vars1' conts1' statics' hp' rd).
Proof.
  intros Hreturns Hagree Hvars Hconts Hstatics Hheaps.
  pose proof
    (same_statics_plain rs (State e1 vars1 conts1 statics hp rd) (rs_next rs)
       (rs_counts rs))
    as Hsame.
  pose proof (same_statics_extends _ _ Hsame) as Hextends.
  exists rconts. simpl.
  split; [ reflexivity | ]. split; [ exact Hreturns | ].
  split; [ exact Hagree | ].
  split; [ exact (env_erased_extend _ _ _ _ Hextends Hvars) | ].
  split; [ exact (conts_erased_extend _ _ _ _ Hextends Hconts) | ].
  split; [ exact (statics_erased_same _ _ _ Hsame Hstatics) | ].
  split; [ exact (heaps_erased_extend _ _ _ _ Hextends Hheaps) | reflexivity ].
Qed.

(* ======================================================================== *)
(* Calls                                                                    *)
(* ======================================================================== *)

(* A closure that is read may be evaluated: a symbol that is bound to
   nothing is no closure, and no location is one. *)
Lemma closure_usable (rs : region_state) (v : value) (called : code_id)
    (slots : list (value_slot * value)) :
  closure_of (live_statics rs) v = Some (called, slots) ->
  value_usable rs v = true.
Proof.
  destruct v as [ | | | s | | | | l | ]; try reflexivity.
  - unfold closure_of, value_usable.
    destruct (lookup s (static_values (live_statics rs)));
      [ intros _; apply orb_true_r | discriminate ].
  - discriminate.
Qed.

(* The fields of a tuple. A tuple that may not be evaluated has no
   field read: a primitive has no value on a symbol bound to nothing,
   and FlambdaPrimitiveSpec.eval_prim, which reads no heap, has none on
   a location. *)
Lemma tuple_fields_erased (rs : region_state) (statics' : static_env)
    (size : nat) (tuple tuple' : value) (fields : list value) :
  symbols_erased rs statics' ->
  value_erased_if_usable rs tuple tuple' ->
  tuple_fields (live_statics rs) size tuple = Some fields ->
  ∃ fields' : list value,
    tuple_fields statics' size tuple' = Some fields'
    ∧ Forall2 (value_erased rs) fields fields'.
Proof.
  intros Hsymbols [_ Htuple]. unfold tuple_fields.
  destruct (value_usable rs tuple) eqn:Husable.
  - exact
      (tuple_fields_from_related (value_erased rs) (live_statics rs) statics'
         size tuple tuple' (value_erased_respects rs statics' Hsymbols)
         (Htuple (or_introl eq_refl)) size 0 fields).
  - destruct size as [ | size];
      [ intros [= <-]; exists []; split; [ reflexivity | constructor ] | ].
    cbn [tuple_fields_from].
    destruct
      (eval_prim (live_statics rs)
         (Unary (Block_load (tuple_access (S size)) Immutable (Z.of_nat 0)))
         [tuple])
      as [v | ] eqn:Hv; [ | discriminate ].
    destruct tuple as [ | | | s | | | | l | ]; try discriminate Husable.
    + pose proof
        (eval_prim_symbols_bound _ _ _ _ s Hv (or_introl eq_refl)) as Hbound.
      unfold value_usable in Husable.
      apply orb_false_iff in Husable as [_ Hnone].
      destruct (lookup s (static_values (live_statics rs)));
        [ discriminate | contradiction ].
    + discriminate Hv.
Qed.

(* The arguments as the code takes them. *)
Lemma arguments_for_code_erased (rs : region_state) (statics' : static_env)
    (c : code) (vs vs' : list value) (args_arity : arity) (ws : list value)
    (args_arity' : arity) :
  symbols_erased rs statics' ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  arguments_for_code (live_statics rs) c vs args_arity
  = Some (ws, args_arity') ->
  ∃ ws' : list value,
    arguments_for_code statics' (erase_code c) vs' args_arity
    = Some (ws', args_arity')
    ∧ Forall2 (value_erased_if_usable rs) ws ws'.
Proof.
  intros Hsymbols Hvs. unfold arguments_for_code.
  rewrite erase_code_is_tupled, erase_code_params_arity.
  destruct (code_is_tupled c);
    [ | intros [= <- <-]; exists vs'; split; [ reflexivity | exact Hvs ] ].
  unfold detuple.
  pose proof (kinds_of_values_if_usable rs vs vs' Hvs) as Hkinds.
  destruct Hvs as [ | tuple tuple' rest rest' Htuple Hrest]; [ discriminate | ].
  destruct args_arity as [ | first rest_arity]; [ discriminate | ].
  rewrite Hkinds. destruct (negb _); [ discriminate | ].
  destruct
    (tuple_fields (live_statics rs) (cardinal_unarized (code_params_arity c))
       tuple)
    as [fields | ] eqn:Hfields; [ | discriminate ].
  destruct
    (tuple_fields_erased rs statics' _ tuple tuple' fields Hsymbols Htuple
       Hfields)
    as (fields' & -> & Hfields').
  intros [= <- <-]. exists (fields' ++ rest').
  split; [ reflexivity | ].
  apply Forall2_app;
    [ exact (values_erased_if_usable_intro rs _ _ Hfields') | exact Hrest ].
Qed.

(* Two calls of one shape: the closures called are related, the
   arguments and those that remain are related where they may be
   evaluated, and the code of the second is that of the first, erased;
   the code has no call that returns or raises to a region parameter. *)
Inductive shape_erased (rs : region_state) : call_shape -> call_shape -> Prop :=
  | Full_call_erased (closure closure' : value) (c : code)
      (vs vs' : list value) (args_arity : arity) :
    value_erased rs closure closure' ->
    code_returns_no_region c = true ->
    Forall2 (value_erased_if_usable rs) vs vs' ->
    shape_erased rs (Full_call closure c vs args_arity)
      (Full_call closure' (erase_code c) vs' args_arity)
  | Over_call_erased (closure closure' : value) (c : code)
      (vs vs' remaining remaining' : list value)
      (args_arity remaining_arity : arity) :
    value_erased rs closure closure' ->
    code_returns_no_region c = true ->
    Forall2 (value_erased_if_usable rs) vs vs' ->
    Forall2 (value_erased_if_usable rs) remaining remaining' ->
    shape_erased rs
      (Over_call closure c vs remaining args_arity remaining_arity)
      (Over_call closure' (erase_code c) vs' remaining' args_arity
         remaining_arity)
  | Partial_call_erased (partial partial' : value) :
    value_erased rs partial partial' ->
    shape_erased rs (Partial_call partial) (Partial_call partial').

(* What the call of a closure is on the machine of regions, the call of
   the closure erased is in the second state. *)
Lemma shape_of_closure_call_erased (rs : region_state)
    (statics' : static_env) (closure closure' : value) (kind : call_kind)
    (vs vs' : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  statics_erased rs statics' ->
  value_erased rs closure closure' ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  shape_of_closure_call (live_statics rs) closure kind vs args_arity
    return_arity
  = Some shape ->
  ∃ shape' : call_shape,
    shape_of_closure_call statics' closure' kind vs' args_arity return_arity
    = Some shape'
    ∧ shape_erased rs shape shape'.
Proof.
  intros (Hcode & Hreturns & Hsymbols & _) Hclosure Hvs.
  unfold shape_of_closure_call.
  destruct (closure_of (live_statics rs) closure) as [[called slots] | ]
    eqn:Hc;
    [ | discriminate ].
  destruct
    (closure_of_erased rs statics' Hsymbols closure closure' called slots
       Hclosure Hc)
    as (slots' & -> & _).
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect => true
     end);
    [ | discriminate ].
  rewrite live_statics_code, Hcode.
  destruct (lookup called (static_code (state_statics (rs_state rs))))
    as [c | ] eqn:Hcalled;
    [ | discriminate ].
  simpl option_map. cbv iota zeta.
  destruct (arguments_for_code (live_statics rs) c vs args_arity)
    as [[ws args_arity'] | ] eqn:Hargs; [ | discriminate ].
  destruct
    (arguments_for_code_erased rs statics' c vs vs' args_arity ws args_arity'
       Hsymbols Hvs Hargs)
    as (ws' & -> & Hws).
  simpl. unfold shape_of_code_call. cbv zeta.
  rewrite erase_code_params_arity, erase_code_result_arity,
    erase_code_is_tupled, (kinds_of_values_if_usable rs ws ws' Hws).
  pose proof (Hreturns called c Hcalled) as Hc_returns.
  destruct (Nat.eqb (length args_arity') (length (code_params_arity c))).
  - intros [= <-]. eexists. split; [ reflexivity | ].
    apply Full_call_erased; assumption.
  - destruct
      (negb
         (kinds_eqb (map kind_of_value ws) (unarize args_arity')
          && partial_or_over_application args_arity' return_arity
               (code_params_arity c) (code_result_arity c)));
      [ discriminate | ].
    destruct
      (Forall2_firstn_skipn (value_erased_if_usable rs)
         (cardinal_unarized (code_params_arity c)) ws ws' Hws)
      as [Hfirst Hrest].
    destruct (Nat.ltb (length (code_params_arity c)) (length args_arity'));
      [ intros [= <-]; eexists; (split; [ reflexivity | ]);
        apply Over_call_erased; assumption
      | ].
    destruct (code_is_tupled c); [ discriminate | ].
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Partial_call_erased, Erased_partial_application;
      [ intros _; exact Hclosure | exact Hws ].
Qed.

Lemma shape_of_closure_call_usable (rs : region_state) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  shape_of_closure_call (live_statics rs) closure kind vs args_arity
    return_arity
  = Some shape ->
  value_usable rs closure = true.
Proof.
  unfold shape_of_closure_call.
  destruct (closure_of (live_statics rs) closure) as [[called slots] | ]
    eqn:Hc;
    [ | discriminate ].
  intros _. exact (closure_usable rs closure called slots Hc).
Qed.

(* The same of a call. The callee is related where it may be evaluated
   or is a location: one that may not be evaluated is no closure, and
   the call has no shape. The closure that a partial application
   keeps is read in the same way. *)
Lemma shape_of_call_erased (rs : region_state) (statics' : static_env)
    (callee callee' : value) (kind : call_kind) (vs vs' : list value)
    (args_arity return_arity : arity) (shape : call_shape) :
  statics_erased rs statics' ->
  value_erased_if_usable rs callee callee' ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  shape_of_call (live_statics rs) callee kind vs args_arity return_arity
  = Some shape ->
  ∃ shape' : call_shape,
    shape_of_call statics' callee' kind vs' args_arity return_arity
    = Some shape'
    ∧ shape_erased rs shape shape'.
Proof.
  intros Hstatics [_ Hcallee] Hvs Hshape.
  assert (Husable : value_usable rs callee = true).
  { unfold shape_of_call in Hshape.
    destruct kind as [named | ];
      [ exact (shape_of_closure_call_usable rs _ _ _ _ _ _ Hshape) | ].
    destruct (partial_application_of callee)
      as [[[closure applied] applied_arity] | ] eqn:Hpartial;
      [ | exact (shape_of_closure_call_usable rs _ _ _ _ _ _ Hshape) ].
    destruct callee; try discriminate Hpartial. reflexivity. }
  specialize (Hcallee (usable_asked rs callee Husable)). revert Hshape.
  unfold shape_of_call.
  destruct kind as [named | ];
    [ apply shape_of_closure_call_erased; assumption | ].
  pose proof (partial_application_erased rs callee callee' Hcallee)
    as Hpartial.
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ].
  - destruct Hpartial as (closure' & applied' & -> & Hclosure & Happlied).
    simpl.
    destruct args_arity as [ | component args_arity]; [ discriminate | ].
    rewrite (kinds_of_values_if_usable rs vs vs' Hvs).
    destruct (kinds_eqb _ _); [ | discriminate ].
    intros Hshape.
    apply (shape_of_closure_call_erased rs statics' closure closure' Indirect
             (applied ++ vs) (applied' ++ vs') _ _ shape Hstatics);
      [ | apply Forall2_app; assumption | exact Hshape ].
    exact
      (Hclosure
         (usable_asked rs closure
            (shape_of_closure_call_usable rs _ _ _ _ _ _ Hshape))).
  - rewrite Hpartial. apply shape_of_closure_call_erased; assumption.
Qed.

(* A full call. The body of the code is run under no continuation that
   [erase] lists: its return and exception continuations stand for what
   the caller's do, which take no region. *)
Lemma call_code_erased (rs : region_state) (statics' : static_env)
    (closure closure' : value) (c : code) (vs vs' : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (hp hp' : heap)
    (rd : read_record) :
  value_erased rs closure closure' ->
  code_returns_no_region c = true ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  cont_takes_no_region return_to ->
  cont_takes_no_region exn_to ->
  cont_erased rs return_to return_to' ->
  cont_erased rs exn_to exn_to' ->
  statics_erased rs statics' ->
  heaps_erased rs hp hp' ->
  step_result_erased rs
    (call_code closure c vs args_arity return_arity return_to exn_to
       (live_statics rs) hp rd)
    (call_code closure' (erase_code c) vs' args_arity return_arity return_to'
       exn_to' statics' hp' rd).
Proof.
  intros Hclosure Hc Hvs Hreturn_takes Hexn_takes Hreturn Hexn Hstatics
    Hheaps.
  unfold step_result_erased, call_code.
  rewrite erase_code_body, erase_binds_statics, erase_code_params_arity,
    erase_code_result_arity, erase_code_params, erase_code_my_closure,
    erase_code_return_continuation, erase_code_exn_continuation,
    (kinds_of_values_if_usable rs vs vs' Hvs).
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [bound | ] eqn:Hbind;
    [ | exact I ].
  destruct
    (bind_params_erased rs (code_params c) vs vs'
       [(code_my_closure c, closure)] [(code_my_closure c, closure')] bound
       Hvs)
    as (bound' & -> & Hbound);
    [ apply env_erased_cons;
      [ intros _; exact Hclosure | apply env_erased_nil ]
    | exact Hbind | ].
  eexists. split; [ reflexivity | ].
  apply state_erased_plain_next.
  - exact (code_body_returns_no_region c Hc).
  - apply conts_agree_call; assumption.
  - exact Hbound.
  - constructor; [ split; [ reflexivity | exact Hreturn ] | ].
    constructor; [ split; [ reflexivity | exact Hexn ] | ].
    constructor.
  - exact Hstatics.
  - exact Hheaps.
Qed.

(* A call of a given shape, given what the two states do with a partial
   application. The full call inside an over-application returns to
   what is left of it. *)
Lemma run_call_erased (rs : region_state) (statics' : static_env)
    (return_partial return_partial' : value -> step_result)
    (shape shape' : call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (hp hp' : heap)
    (rd : read_record) :
  shape_erased rs shape shape' ->
  cont_takes_no_region return_to ->
  cont_takes_no_region exn_to ->
  cont_erased rs return_to return_to' ->
  cont_erased rs exn_to exn_to' ->
  statics_erased rs statics' ->
  heaps_erased rs hp hp' ->
  (∀ partial partial' : value,
    value_erased rs partial partial' ->
    step_result_erased rs (return_partial partial)
      (return_partial' partial')) ->
  step_result_erased rs
    (run_call return_partial (Some shape) return_arity return_to exn_to
       (live_statics rs) hp rd)
    (run_call return_partial' (Some shape') return_arity return_to' exn_to'
       statics' hp' rd).
Proof.
  intros Hshape Hreturn_takes Hexn_takes Hreturn Hexn Hstatics Hheaps
    Hpartial.
  destruct Hshape
    as [ closure closure' c vs vs' args_arity Hclosure Hc Hvs
       | closure closure' c vs vs' remaining remaining' args_arity
           remaining_arity Hclosure Hc Hvs Hremaining
       | partial partial' Hp ];
    simpl.
  - apply call_code_erased; assumption.
  - rewrite erase_code_result_arity.
    exact
      (call_code_erased rs statics' closure closure' c vs vs' args_arity
         (code_result_arity c)
         (Over_application remaining remaining_arity return_arity return_to
            exn_to)
         (Over_application remaining' remaining_arity return_arity return_to'
            exn_to')
         exn_to exn_to' hp hp' rd Hclosure Hc Hvs I Hexn_takes
         (Erased_over_application rs remaining remaining' remaining_arity
            return_arity return_to return_to' exn_to exn_to' Hremaining
            Hreturn_takes Hexn_takes Hreturn Hexn)
         Hexn Hstatics Hheaps).
  - exact (Hpartial partial partial' Hp).
Qed.

(* A call, given what the two states do when the return continuation is
   applied to a partial application. *)
Lemma call_function_erased_from (rs : region_state) (statics' : static_env)
    (callee callee' : value) (kind : call_kind) (vs vs' : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (hp hp' : heap)
    (rd : read_record) :
  value_erased_if_usable rs callee callee' ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  cont_takes_no_region return_to ->
  cont_takes_no_region exn_to ->
  cont_erased rs return_to return_to' ->
  cont_erased rs exn_to exn_to' ->
  statics_erased rs statics' ->
  heaps_erased rs hp hp' ->
  (∀ partial partial' : value,
    value_erased rs partial partial' ->
    step_result_erased rs
      (apply_cont_value return_to [partial] (live_statics rs) hp rd)
      (apply_cont_value return_to' [partial'] statics' hp' rd)) ->
  step_result_erased rs
    (call_function callee kind vs args_arity return_arity return_to exn_to
       (live_statics rs) hp rd)
    (call_function callee' kind vs' args_arity return_arity return_to'
       exn_to' statics' hp' rd).
Proof.
  intros Hcallee Hvs Hreturn_takes Hexn_takes Hreturn Hexn Hstatics Hheaps
    Hpartial.
  unfold call_function.
  destruct
    (shape_of_call (live_statics rs) callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | exact I ].
  destruct
    (shape_of_call_erased rs statics' callee callee' kind vs vs' args_arity
       return_arity shape Hstatics Hcallee Hvs Hshape)
    as (shape' & -> & Hshapes).
  exact
    (run_call_erased rs statics' _ _ shape shape' return_arity _ _ _ _ hp
       hp' rd Hshapes Hreturn_takes Hexn_takes Hreturn Hexn Hstatics Hheaps
       Hpartial).
Qed.

(* Applying a continuation that takes no region. The values are related
   where they may be evaluated or are locations: a continuation with
   extra arguments gives those it has kept since the call. What is
   left of an over-application calls the value it is given, which is
   no closure if it may not be evaluated. *)
Lemma apply_cont_value_erased (rs : region_state) (statics' : static_env)
    (c c' : cont_value) :
  cont_erased rs c c' ->
  cont_takes_no_region c ->
  statics_erased rs statics' ->
  ∀ (vs vs' : list value) (hp hp' : heap) (rd : read_record),
    Forall2 (value_erased_if_usable rs) vs vs' ->
    heaps_erased rs hp hp' ->
    step_result_erased rs (apply_cont_value c vs (live_statics rs) hp rd)
      (apply_cont_value c' vs' statics' hp' rd).
Proof.
  intros Hc.
  induction Hc
    as [ k
       | rconts [params metadata handler] vars vars' conts conts' recursive
           Hh Hrecursive Hagree Hvars Hconts
       | c c' extra extra' Htakes Hc IH Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn_takes Hexn_takes
           Hreturn IHreturn Hexn _ ];
    intros Htop Hstatics vs vs' hp hp' rd Hvs Hheaps.
  - simpl. exists vs', hp'. split; [ reflexivity | ].
    split; [ exact Hheaps | exact Hvs ].
  - simpl in Htop.
    rewrite (erase_handler_no_region _ params metadata handler Htop).
    cbn [apply_cont_value].
    rewrite (kinds_of_values_if_usable rs vs vs' Hvs).
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct
      (bind_params_erased rs params vs vs' vars vars' bound Hvs Hvars Hbind)
      as (bound' & -> & Hbound).
    eexists. split; [ reflexivity | ].
    apply state_erased_plain_next.
    + exact Hh.
    + exact
        (conts_agree_rec rconts conts vars conts recursive recursive Hagree).
    + exact Hbound.
    + apply rec_closures_erased; assumption.
    + exact Hstatics.
    + exact Hheaps.
  - cbn [apply_cont_value].
    apply IH;
      [ exact Htakes | exact Hstatics | apply Forall2_app; assumption
      | exact Hheaps ].
  - rewrite !apply_cont_value_over_application.
    destruct Hvs as [ | v v' vs vs' Hv Hvs]; [ exact I | ].
    destruct Hvs as [ | w w' vs vs' Hw Hvs]; [ | exact I ].
    apply call_function_erased_from; try assumption.
    intros partial partial' Hpartial.
    apply IHreturn; [ exact Hreturn_takes | exact Hstatics | | exact Hheaps ].
    constructor; [ | constructor ].
    exact (value_erased_if_usable_intro rs _ _ Hpartial).
Qed.

(* A call. *)
Lemma call_function_erased (rs : region_state) (statics' : static_env)
    (callee callee' : value) (kind : call_kind) (vs vs' : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (hp hp' : heap)
    (rd : read_record) :
  value_erased_if_usable rs callee callee' ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  cont_takes_no_region return_to ->
  cont_takes_no_region exn_to ->
  cont_erased rs return_to return_to' ->
  cont_erased rs exn_to exn_to' ->
  statics_erased rs statics' ->
  heaps_erased rs hp hp' ->
  step_result_erased rs
    (call_function callee kind vs args_arity return_arity return_to exn_to
       (live_statics rs) hp rd)
    (call_function callee' kind vs' args_arity return_arity return_to'
       exn_to' statics' hp' rd).
Proof.
  intros Hcallee Hvs Hreturn_takes Hexn_takes Hreturn Hexn Hstatics Hheaps.
  apply call_function_erased_from; try assumption.
  intros partial partial' Hpartial.
  apply (apply_cont_value_erased rs statics' return_to return_to' Hreturn
           Hreturn_takes Hstatics);
    [ | exact Hheaps ].
  constructor; [ | constructor ].
  exact (value_erased_if_usable_intro rs _ _ Hpartial).
Qed.

(* ======================================================================== *)
(* A step of [step]                                                         *)
(* ======================================================================== *)

(* A [Let] at whose head the machine of regions does nothing of its
   own is erased to itself with its body erased, and the simples that
   its step evaluates are those of its defining expression. *)
Lemma head_plain_let (rconts : region_conts) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  head_plain (Let x kind d body) = true ->
  erase rconts (Let x kind d body) = Let x kind d (erase rconts body)
  ∧ evaluated_simples (Let x kind d body) = simples_of_named d.
Proof.
  destruct d as [ s | p args | | | | | ]; try (intros _; split; reflexivity).
  destruct p; try (intros _; split; reflexivity). discriminate.
Qed.

Lemma find_arm_erased (rconts : region_conts) (i : Z)
    (arms : list (Z * apply_cont)) (ac : apply_cont) :
  find_arm i arms = Some ac ->
  find_arm i (map (fun '(i0, ac0) => (i0, erase_apply_cont rconts ac0)) arms)
  = Some (erase_apply_cont rconts ac).
Proof.
  induction arms as [ | [d ac0] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb i d); [ intros [= <-]; reflexivity | exact IH ].
Qed.

(* A continuation applied to values none of which is of the kind
   [Region] has no step, or takes no region: the kinds of the values
   are those of the parameters of its handler. *)
Lemma applied_takes_no_region (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  Forall (fun v => kind_of_value v <> Region) vs ->
  apply_cont_value c vs statics hp rd = Undefined ∨ cont_takes_no_region c.
Proof.
  intros Hkinds.
  destruct c
    as [ [params metadata handler] vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to ];
    try (right; exact I).
  cbn [apply_cont_value].
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
    eqn:Hk;
    [ | left; reflexivity ].
  right. apply kinds_eqb_eq in Hk.
  unfold cont_takes_no_region, handler_takes_no_region, region_params.
  simpl. rewrite <- Hk. clear Hk. apply negb_true_iff.
  induction Hkinds as [ | v vs Hv _ IH]; simpl; [ reflexivity | ].
  rewrite IH, orb_false_r.
  destruct (kind_eqb (kind_of_value v) Region) eqn:Hkind; [ | reflexivity ].
  apply kind_eqb_eq in Hkind. contradiction.
Qed.

(* One step of [step], under the static environment of the frames, at
   an expression at whose head the machine of regions does nothing of
   its own and that is not a [Let_static]. The two states have heaps
   related ([heaps_erased]): a [Let] runs its defining expression on
   each, and what it adds to them leaves them related
   (FlambdaRegionsErasedProofs.exec_named_erased). The premises beside
   the relation: the expression names no symbol that the machine drew;
   the simples that the step evaluates may be evaluated; and at an
   [Apply_cont] no value of an argument is of the kind [Region]. *)
Lemma step_erased (rs : region_state) (rconts : region_conts) (e : expr)
    (vars vars' : env) (conts conts' : cont_env) (statics' : static_env)
    (hp hp' : heap) (rd : read_record) :
  calls_return_no_region rconts e = true ->
  conts_agree rconts conts ->
  env_erased rs vars vars' ->
  conts_erased rs conts conts' ->
  statics_erased rs statics' ->
  heaps_erased rs hp hp' ->
  head_plain e = true ->
  (∀ (bindings : list static_binding) (body : expr),
    e <> Let_static bindings body) ->
  expr_below (rs_base rs) e = true ->
  simples_usable rs vars (evaluated_simples e) ->
  (∀ (ac : apply_cont) (vs : list value),
    e = Apply_cont ac ->
    eval_simples vars (ac_args ac) = Some vs ->
    Forall (fun v => kind_of_value v <> Region) vs) ->
  step_result_erased rs (step (State e vars conts (live_statics rs) hp rd))
    (step (State (erase rconts e) vars' conts' statics' hp' rd)).
Proof.
  intros Hreturns Hagree Hvars Hconts Hstatics Hheaps Hhead Hnot_static
    Hbelow Husable Hkinds.
  pose proof (proj1 (proj2 (proj2 Hstatics))) as Hsymbols.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - (* A [Let]. *)
    destruct (head_plain_let rconts x xkind d body Hhead) as [-> Hsimples].
    rewrite Hsimples in Husable. simpl in Hbelow.
    apply andb_true_iff in Hbelow as [Hd_below _].
    rewrite !step_let_exec.
    destruct (exec_named (live_statics rs) hp rd vars d)
      as [[[v hp1] rd1] | ] eqn:Hv;
      [ | exact I ].
    destruct
      (exec_named_erased rs statics' vars vars' d hp hp' hp1 rd rd1 v
         Hsymbols Hvars Hd_below Husable Hheaps Hv)
      as (v' & hp1' & -> & Hrelated & Hheaps1).
    rewrite (kind_of_value_erased rs v v' Hrelated).
    destruct (kind_eqb (kind_of_value v) xkind); [ | exact I ].
    eexists. split; [ reflexivity | ].
    apply state_erased_plain_next; try assumption.
    apply env_erased_cons; [ intros _; exact Hrelated | exact Hvars ].
  - (* A [Let_static]: not here. *)
    contradiction (Hnot_static bindings body eq_refl).
  - (* A non-recursive [Let_cont]. *)
    cbn [erase]. rewrite !step_let_cont_non_recursive.
    simpl in Hreturns. apply andb_true_iff in Hreturns as [Hh Hbody].
    eexists. split; [ reflexivity | ].
    apply state_erased_plain_next; try assumption.
    + exact (conts_agree_cons rconts conts k h vars conts [] Hagree).
    + constructor; [ | exact Hconts ].
      split; [ reflexivity | ].
      exact
        (Erased_handler rs rconts h vars vars' conts conts' [] Hh eq_refl
           Hagree Hvars Hconts).
  - (* A recursive [Let_cont]. *)
    change (erase rconts (Let_cont_recursive handlers body))
      with
        (Let_cont_recursive
           (erase_handlers (region_conts_of handlers ++ rconts) handlers)
           (erase (region_conts_of handlers ++ rconts) body)).
    rewrite !step_let_cont_recursive, erase_handlers_bind_statics.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    simpl in Hreturns. apply andb_true_iff in Hreturns as [Hhandlers Hbody].
    eexists. split; [ reflexivity | ].
    apply state_erased_plain_next; try assumption.
    + exact
      (conts_agree_rec rconts conts vars conts handlers handlers Hagree).
    + apply rec_closures_erased; assumption.
  - (* A call. *)
    cbn [erase]. rewrite !step_apply.
    simpl in Hbelow, Hreturns.
    apply andb_true_iff in Hbelow as [Hcallee_below Hbelow].
    rewrite forallb_app in Hbelow.
    apply andb_true_iff in Hbelow as [Hargs_below Hextra_below].
    apply andb_true_iff in Hreturns as [Hreturn Hexn].
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra;
      [ | exact I ].
    destruct
      (eval_simple_erased rs vars vars' (ap_callee ap) callee Hvars
         Hcallee_below Hcallee
         (Husable (ap_callee ap) callee (or_introl eq_refl) Hcallee))
      as (callee' & -> & Hcallee').
    destruct
      (eval_simples_erased rs vars vars' (ap_args ap) Hvars Hargs_below
         (fun s v Hin =>
            Husable s v (or_intror (in_or_app _ _ _ (or_introl Hin))))
         vs Hargs)
      as (vs' & -> & Hvs).
    destruct
      (eval_simples_erased rs vars vars' (exn_extra_simples ap) Hvars
         Hextra_below
         (fun s v Hin =>
            Husable s v (or_intror (in_or_app _ _ _ (or_intror Hin))))
         extra Hextra)
      as (extra' & -> & Hextra').
    apply call_function_erased.
    + exact (value_erased_if_usable_intro rs _ _ Hcallee').
    + exact (values_erased_if_usable_intro rs _ _ Hvs).
    + exact (takes_no_region_resolve rconts conts _ Hagree Hreturn).
    + apply with_extra_args_takes_no_region.
      exact (takes_no_region_resolve rconts conts _ Hagree Hexn).
    + apply resolve_erased, Hconts.
    + apply with_extra_args_erased.
      * exact (takes_no_region_resolve rconts conts _ Hagree Hexn).
      * apply resolve_erased, Hconts.
      * exact (values_erased_if_usable_intro rs _ _ Hextra').
    + exact Hstatics.
    + exact Hheaps.
  - (* An [Apply_cont]. *)
    cbn [erase]. rewrite !step_apply_cont. simpl in Hbelow.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    unfold apply_continuation.
    destruct
      (applied_takes_no_region (resolve (ac_continuation ac) conts) vs
         (live_statics rs) hp rd (Hkinds ac vs eq_refl Hargs))
      as [Hundefined | Htakes];
      [ rewrite Hundefined; exact I | ].
    rewrite (erase_apply_cont_no_region rconts conts ac Hagree Htakes).
    destruct
      (eval_simples_erased rs vars vars' (ac_args ac) Hvars Hbelow Husable vs
         Hargs)
      as (vs' & -> & Hvs).
    apply apply_cont_value_erased;
      [ apply resolve_erased, Hconts | exact Htakes | exact Hstatics
      | | exact Hheaps ].
    exact (values_erased_if_usable_intro rs _ _ Hvs).
  - (* A [Switch]. *)
    cbn [erase]. rewrite !step_switch. simpl in Hbelow.
    apply andb_true_iff in Hbelow as [Hscrutinee_below _].
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
    destruct
      (eval_simple_erased rs vars vars' scrutinee v Hvars Hscrutinee_below Hv
         (Husable scrutinee v (or_introl eq_refl) Hv))
      as (v' & -> & Hrelated).
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
    rewrite (naked_immediate_erased rs v v' i Hrelated Hi).
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    rewrite (find_arm_erased rconts i arms ac Harm).
    eexists. split; [ reflexivity | ].
    exact
      (state_erased_plain_next rs rconts (Apply_cont ac) vars vars' conts
         conts' _ statics' hp hp' rd eq_refl Hagree Hvars Hconts Hstatics
         Hheaps).
  - (* [Invalid]. *)
    rewrite step_invalid. exact I.
Qed.

(* ======================================================================== *)
(* A step of the machine of regions                                         *)
(* ======================================================================== *)

(* What the second state answers to a result of the machine of
   regions: nothing is asked if it has no step; an exit is answered by
   an exit through the same continuation, with the values related
   where they may be evaluated or are locations, under the state that
   the first left by; a step, by a step to a state related. *)
Definition result_erased (result : region_step_result)
    (result' : step_result) : Prop :=
  match result with
  | Region_undefined _ => True
  | Region_exit k vs rs1 =>
    ∃ (vs' : list value) (hp : heap) (rd : read_record),
      result' = Exit k vs' hp rd ∧ Forall2 (value_erased_if_usable rs1) vs vs'
  | Region_step rs1 =>
    ∃ s1' : state, result' = Step s1' ∧ state_erased rs1 s1'
  end.

(* The step of [step] under the static environment of the frames. *)
Theorem plain_step_erased (rs : region_state) (s' : state) :
  state_erased rs s' ->
  head_plain (state_expr (rs_state rs)) = true ->
  (∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) <> Let_static bindings body) ->
  expr_below (rs_base rs) (state_expr (rs_state rs)) = true ->
  simples_usable rs (state_vars (rs_state rs))
    (evaluated_simples (state_expr (rs_state rs))) ->
  (∀ (ac : apply_cont) (vs : list value),
    state_expr (rs_state rs) = Apply_cont ac ->
    eval_simples (state_vars (rs_state rs)) (ac_args ac) = Some vs ->
    Forall (fun v => kind_of_value v <> Region) vs) ->
  result_erased (plain_step rs) (step s').
Proof.
  intros
    (rconts & He & Hreturns & Hagree & Hvars & Hconts & Hstatics & Hheap
     & Hreads)
    Hhead Hnot_static Hbelow Husable Hkinds.
  destruct s' as [e' vars' conts' statics' hp' rd'].
  simpl in He, Hvars, Hconts, Hstatics, Hheap, Hreads. subst e' rd'.
  pose proof
    (step_erased rs rconts (state_expr (rs_state rs))
       (state_vars (rs_state rs)) vars' (state_conts (rs_state rs)) conts'
       statics' (state_heap (rs_state rs)) hp' (state_reads (rs_state rs))
       Hreturns Hagree Hvars Hconts Hstatics Hheap Hhead Hnot_static Hbelow
       Husable Hkinds)
    as Hcore.
  unfold plain_step, with_statics.
  destruct
    (step
       (State (state_expr (rs_state rs)) (state_vars (rs_state rs))
          (state_conts (rs_state rs)) (live_statics rs)
          (state_heap (rs_state rs)) (state_reads (rs_state rs))))
    as [s1 | k vs hp rd | ];
    simpl in Hcore |- *.
  - exact Hcore.
  - destruct Hcore as (vs' & hp1' & -> & _ & Hvs). exists vs', hp1', rd.
    split; [ reflexivity | exact Hvs ].
  - exact I.
Qed.

(* The step of a [Let_static], on the static environment of the
   state. *)
Theorem static_step_erased (rs : region_state) (s' : state)
    (bindings : list static_binding) (body : expr) :
  state_erased rs s' ->
  state_expr (rs_state rs) = Let_static bindings body ->
  expr_below (rs_base rs) (state_expr (rs_state rs)) = true ->
  simples_usable rs (state_vars (rs_state rs))
    (evaluated_simples (state_expr (rs_state rs))) ->
  result_erased (static_step rs) (step s').
Proof.
  intros
    (rconts & He & Hreturns & Hagree & Hvars & Hconts & Hstatics & Hheap
     & Hreads)
    Hexpr Hbelow Husable.
  destruct s' as [e' vars' conts' statics' hp' rd'].
  simpl in He, Hvars, Hconts, Hstatics, Hheap, Hreads. subst e' rd'.
  rewrite Hexpr in Hreturns, Hbelow, Husable |- *.
  simpl in Hreturns, Hbelow, Husable.
  apply andb_true_iff in Hreturns as [Hbindings_return Hbody].
  apply andb_true_iff in Hbelow as [Hbindings_below _].
  pose proof
    (eval_static_bindings_erased (state_vars (rs_state rs)) vars' bindings rs
       statics')
    as Hgroup.
  unfold static_step. cbn [erase].
  destruct (rs_state rs) as [e vars conts statics hp rd] eqn:Hstate.
  simpl in Hexpr, Hgroup, Hvars, Hconts, Hagree, Husable. subst e.
  rewrite !step_let_static.
  destruct (eval_static_bindings vars bindings statics) as [statics1 | ]
    eqn:Heval;
    [ | exact I ].
  destruct
    (Hgroup statics1 Hvars Hstatics Hbindings_below Hbindings_return Husable
       eq_refl)
    as (statics1' & -> & Hstatics1 & Hextends).
  eexists. split; [ reflexivity | ].
  assert (Hsame :
    same_statics (with_region_statics rs statics1)
      (Region_state (State body vars conts statics1 hp rd) (rs_frames rs)
         (rs_base rs) (rs_next rs) (rs_counts rs) (rs_cells rs))).
  { apply same_statics_intro; reflexivity. }
  pose proof
    (region_extends_trans _ _ _ Hextends (same_statics_extends _ _ Hsame))
    as Hextends1.
  exists rconts. simpl.
  split; [ reflexivity | ]. split; [ exact Hbody | ].
  split; [ exact Hagree | ].
  split; [ exact (env_erased_extend _ _ _ _ Hextends1 Hvars) | ].
  split; [ exact (conts_erased_extend _ _ _ _ Hextends1 Hconts) | ].
  split; [ exact (statics_erased_same _ _ _ Hsame Hstatics1) | ].
  split; [ exact (heaps_erased_extend _ _ _ _ Hextends1 Hheap) | reflexivity ].
Qed.

(* ======================================================================== *)
(* A step of [region_step] at such an expression                            *)
(* ======================================================================== *)

(* Where the test of the simples passes, each simple that the step
   evaluates may be evaluated, unless it is an open mark at the place
   of a region: with no place of a region, and with no open mark. *)
Lemma simples_fault_no_place (rs : region_state) (simples : list simple) :
  simples_fault rs simples [] = None ->
  forallb (simple_usable rs) simples = true.
Proof.
  induction simples as [ | s simples IH]; simpl; [ reflexivity | ].
  destruct (simple_usable rs s); [ exact IH | ].
  destruct (simple_open_mark rs s); discriminate.
Qed.

Lemma simples_fault_no_mark (rs : region_state) (simples : list simple) :
  ∀ places : list bool,
    simples_fault rs simples places = None ->
    existsb (simple_open_mark rs) simples = false ->
    forallb (simple_usable rs) simples = true.
Proof.
  induction simples as [ | s simples IH]; intros places; simpl;
    [ reflexivity | ].
  destruct (simple_usable rs s).
  - intros Hfault Hmarks. apply orb_false_iff in Hmarks as [_ Hmarks].
    exact (IH _ Hfault Hmarks).
  - destruct (simple_open_mark rs s);
      [ intros _ Hmarks; discriminate Hmarks | discriminate ].
Qed.

(* The counts change nothing that the tests read. *)
Lemma count_uses_with_counts (rs : region_state) :
  ∃ counts : region_counts, count_uses rs = with_counts rs counts.
Proof. eexists. reflexivity. Qed.

Lemma simple_open_mark_with_counts (rs : region_state)
    (counts : region_counts) (s : simple) :
  simple_open_mark (with_counts rs counts) s = simple_open_mark rs s.
Proof. Transparent live_statics. reflexivity. Opaque live_statics. Qed.

Lemma existsb_open_mark_with_counts (rs : region_state)
    (counts : region_counts) (simples : list simple) :
  existsb (simple_open_mark (with_counts rs counts)) simples
  = existsb (simple_open_mark rs) simples.
Proof.
  induction simples as [ | s simples IH]; simpl; [ reflexivity | ].
  rewrite simple_open_mark_with_counts, IH. reflexivity.
Qed.

(* A step of the machine of regions at an expression at whose head it
   does nothing of its own, and that is not an [Apply_cont] one of
   whose arguments is an open mark: the step is that of [plain_step],
   or of [static_step], after the tests. The second state answers it
   with one step of [step]. *)
Theorem region_step_shared (rs : region_state) (s' : state) :
  state_erased rs s' ->
  head_plain (state_expr (rs_state rs)) = true ->
  (∀ ac : apply_cont,
    state_expr (rs_state rs) = Apply_cont ac ->
    existsb (simple_open_mark rs) (ac_args ac) = false) ->
  expr_below (rs_base rs) (state_expr (rs_state rs)) = true ->
  (∀ (ac : apply_cont) (vs : list value),
    state_expr (rs_state rs) = Apply_cont ac ->
    eval_simples (state_vars (rs_state rs)) (ac_args ac) = Some vs ->
    Forall (fun v => kind_of_value v <> Region) vs) ->
  result_erased (region_step rs) (step s').
Proof.
  intros Hstate Hhead Hmarks Hbelow Hkinds.
  unfold region_step. cbv zeta.
  destruct
    (simples_fault rs (evaluated_simples (state_expr (rs_state rs)))
       (region_places rs))
    as [fault | ] eqn:Hfault;
    [ exact I | ].
  assert (Husable :
    forallb (simple_usable rs)
      (evaluated_simples (state_expr (rs_state rs)))
    = true).
  { revert Hfault. unfold region_places.
    destruct (state_expr (rs_state rs)) eqn:He;
      try exact (simples_fault_no_place rs _).
    intros Hfault.
    exact (simples_fault_no_mark rs _ _ Hfault (Hmarks _ eq_refl)). }
  destruct (count_uses_with_counts rs) as [counts Hcounts]. rewrite Hcounts.
  pose proof
    (state_erased_same rs (with_counts rs counts) s'
       (same_statics_with_counts rs counts) eq_refl Hstate)
    as Hstate_c.
  assert (Husable_c :
    simples_usable (with_counts rs counts) (state_vars (rs_state rs))
      (evaluated_simples (state_expr (rs_state rs)))).
  { intros s v Hin Hs.
    rewrite
      (value_usable_extends rs (with_counts rs counts) v
         (same_statics_extends _ _ (same_statics_with_counts rs counts))).
    exact (simples_usable_of_test rs _ Husable s v Hin Hs). }
  assert (Hplain :
    (∀ (bindings : list static_binding) (body : expr),
      state_expr (rs_state rs) <> Let_static bindings body) ->
    result_erased (plain_step (with_counts rs counts)) (step s')).
  { intros Hnot_static.
    exact
      (plain_step_erased (with_counts rs counts) s' Hstate_c Hhead
         Hnot_static Hbelow Husable_c Hkinds). }
  assert (Hstatic :
    ∀ (bindings : list static_binding) (body : expr),
      state_expr (rs_state rs) = Let_static bindings body ->
      result_erased (static_step (with_counts rs counts)) (step s')).
  { intros bindings body He.
    exact
      (static_step_erased (with_counts rs counts) s' bindings body Hstate_c
         He Hbelow Husable_c). }
  destruct (state_expr (rs_state rs))
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - (* A [Let]: after the test of an allocation on the heap and the test
       of a store, the step is that of [plain_step]. *)
    assert (Hnot_static :
      ∀ (bindings : list static_binding) (body0 : expr),
        Let x xkind d body <> Let_static bindings body0)
      by (intros bindings body0; discriminate).
    destruct d as [ s | p args | | | | | ];
      try (destruct p; try discriminate Hhead);
      cbv beta iota; simpl in Hhead;
      (destruct (marker_of x body); try discriminate Hhead);
      cbv beta iota;
      (destruct ((is_allocation _ || is_cell_allocation _) && existsb _ _);
       [ exact I | ]);
      (destruct (store_fault _ _); [ exact I | exact (Hplain Hnot_static) ]).
  - (* A [Let_static]. *)
    destruct
      (existsb (simple_local (with_counts rs counts))
         (evaluated_simples (Let_static bindings body)));
      [ exact I | ].
    exact (Hstatic bindings body eq_refl).
  - apply Hplain. intros bindings body0. discriminate.
  - apply Hplain. intros bindings body0. discriminate.
  - apply Hplain. intros bindings body0. discriminate.
  - (* An [Apply_cont]. *)
    rewrite existsb_open_mark_with_counts, (Hmarks ac eq_refl).
    apply Hplain. intros bindings body0. discriminate.
  - apply Hplain. intros bindings body0. discriminate.
  - apply Hplain. intros bindings body0. discriminate.
Qed.
