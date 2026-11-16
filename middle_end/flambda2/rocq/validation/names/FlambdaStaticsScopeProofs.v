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


(* Proofs of the statements of FlambdaStaticsScopeSpec.v. Nothing here
   needs review.

   Each rewrite is shown to give a term that keeps the scope of the
   term it is given (FlambdaScopedProofs.scoped), by recursion on the
   term with the lemmas of that file for each construct. A group whose
   bindings are permuted has the same closed code and the same free
   names; a group with bindings removed has no more of either. The
   lemmas for FlambdaUnaliasParamsSpec.unalias_params are in
   FlambdaUnaliasParamsNamesProofs.v. *)

From Stdlib Require Import ZArith List Bool Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaUnusedStaticsSpec FlambdaUnusedHeadStaticsSpec
  FlambdaStaticsOrderSpec
  FlambdaMyClosureSpec FlambdaWellFormedSpec FlambdaLiftSpec
  FlambdaLiftFromCodeSpec FlambdaLiftFromArmsSpec FlambdaLiftAroundSpec
  FlambdaLiftObservedSpec FlambdaCopyCodeSpec
  FlambdaHoistStaticsSpec
  FlambdaUnusedSlotsSpec FlambdaKnownSymbolsSpec FlambdaKnownClosuresSpec
  FlambdaKnownSlotsSpec FlambdaStubCallsSpec FlambdaStaticsInScopeSpec
  FlambdaFloatCodeSpec FlambdaGroupsUpSpec FlambdaUnaliasParamsSpec
  FlambdaArmNumbersSpec FlambdaValidateSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaKeepsScopeProofs FlambdaStaticsScopeSpec.
From OxCamlFlambda2 Require Import FlambdaUnaliasRecParamsSpec.
From OxCamlFlambda2 Require FlambdaUnaliasRecParamsNamesProofs.
From OxCamlFlambda2 Require FlambdaUnaliasParamsNamesProofs
  FlambdaWrapReturnsNamesProofs FlambdaUntagParamsNamesProofs
  FlambdaGiveScrutineeNamesProofs FlambdaArmNumbersNamesProofs
  FlambdaShadowParamsNamesProofs FlambdaTwinParamsNamesProofs
  FlambdaBindConstParamsNamesProofs.
Import ListNotations.

(* ======================================================================== *)
(* Groups of bindings, and a [Let] added                                    *)
(* ======================================================================== *)

(* The same bindings in another order. *)
Lemma scoped_bindings_permutation (bindings bindings' : list static_binding) :
  Permutation bindings bindings' -> scoped_bindings bindings bindings'.
Proof.
  intros Hperm Hclosed.
  assert (Hin : ∀ binding : static_binding,
            In binding bindings' -> In binding bindings).
  { intros binding. apply Permutation_in, Permutation_sym, Hperm. }
  split; [ | split ].
  - apply forallb_forall. intros binding Hbinding.
    exact (proj1 (forallb_forall _ _) Hclosed binding (Hin binding Hbinding)).
  - intros x Hx. apply in_flat_map in Hx. destruct Hx as (binding & Hb & Hx).
    apply in_flat_map. exists binding. split; [ apply Hin, Hb | exact Hx ].
  - intros k Hk. apply in_flat_map in Hk. destruct Hk as (binding & Hb & Hk).
    apply in_flat_map. exists binding. split; [ apply Hin, Hb | exact Hk ].
Qed.

(* A binding removed from the front. *)
Lemma scoped_bindings_drop (binding : static_binding)
    (bindings bindings' : list static_binding) :
  scoped_bindings bindings bindings' ->
  scoped_bindings (binding :: bindings) bindings'.
Proof.
  intros Hbindings Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  apply andb_true_iff in Hclosed. destruct Hclosed as [_ Hrest].
  destruct (Hbindings Hrest) as (Hrest' & Hv & Hk).
  split; [ exact Hrest' | split; apply incl_appr; assumption ].
Qed.

(* A symbol bound to a constant with no more variables. *)
Lemma scoped_bindings_cons_symbol (s : symbol) (const const' : static_const)
    (bindings bindings' : list static_binding) :
  incl (free_variables_of_static_const const')
    (free_variables_of_static_const const) ->
  scoped_bindings bindings bindings' ->
  scoped_bindings (Bind_symbol s const :: bindings)
    (Bind_symbol s const' :: bindings').
Proof.
  intros Hconst Hbindings Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  apply andb_true_iff in Hclosed. destruct Hclosed as [_ Hrest].
  destruct (Hbindings Hrest) as (Hrest' & Hv & Hk).
  scope. rewrite Hrest'.
  split; [ reflexivity | ].
  split; (apply incl_app_app; [ | assumption ]);
    [ exact Hconst | apply incl_refl ].
Qed.

(* A [Let] of something with no variable put before a term. *)
Lemma scoped_let_closed (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  free_variables_of_named d = [] -> scoped body (Let x kind d body).
Proof.
  intros Hd Hclosed. scope. rewrite Hd.
  split; [ exact Hclosed | split; [ | apply incl_refl ] ].
  intros y Hy. apply in_without in Hy. exact (proj1 Hy).
Qed.

(* ======================================================================== *)
(* The removal of static bindings                                           *)
(* ======================================================================== *)

Fixpoint erase_expr_scoped (D : dropped) (e : expr) {struct e} :
  scoped e (erase_expr D e)
with erase_handler_scoped (D : dropped) (h : continuation_handler)
    {struct h} :
  scoped_handler h (erase_handler D h)
with erase_code_scoped (D : dropped) (c : code) {struct c} :
  scoped_code c (erase_code D c).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [erase_expr]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply erase_expr_scoped ].
    + apply scoped_let_static; [ | apply erase_expr_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [keep_map];
        [ apply scoped_bindings_nil | ].
      destruct (binding_kept D binding);
        [ | apply scoped_bindings_drop, IH ].
      destruct binding as [id c | id | sym const]; cbn [erase_binding].
      * apply scoped_bindings_cons_code;
          [ apply erase_code_scoped | exact IH ].
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_same, IH.
    + apply scoped_let_cont;
        [ apply erase_handler_scoped | apply erase_expr_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply erase_expr_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons; [ apply erase_handler_scoped | exact IH ].
  - destruct h as [params metadata handler]. cbn [erase_handler].
    apply scoped_handler_intro, erase_expr_scoped.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [erase_code]. apply scoped_code_intro, erase_expr_scoped.
Qed.

Theorem erase_expr_keeps_scope : erase_expr_keeps_scope_statement.
Proof.
  intros D. apply keeps_scope_intro. intros e. apply erase_expr_scoped.
Qed.

Theorem without_statics_keeps_scope : without_statics_keeps_scope_statement.
Proof.
  intros unused. apply keeps_scope_intro. intros e.
  unfold without_statics, drop_statics.
  destruct (expr_avoids unused e); [ | apply scoped_refl ].
  eapply scoped_trans; [ apply erase_expr_scoped | ].
  eapply scoped_trans;
    [ apply drop_deleted_code_scoped | apply merge_statics_scoped ].
Qed.

(* ======================================================================== *)
(* The order of static bindings                                             *)
(* ======================================================================== *)

Lemma to_front_permutation (is_it : static_binding -> bool)
    (bindings : list static_binding) :
  Permutation bindings (to_front is_it bindings).
Proof.
  induction bindings as [ | binding rest IH]; cbn [to_front];
    [ apply perm_nil | ].
  destruct (is_it binding); [ apply Permutation_refl | ].
  destruct (to_front is_it rest) as [ | found rest'];
    [ apply Permutation_refl | ].
  destruct (is_it found); [ | apply Permutation_refl ].
  eapply perm_trans; [ apply perm_skip, IH | apply perm_swap ].
Qed.

Lemma order_as_permutation (pattern bindings : list static_binding) :
  Permutation bindings (order_as pattern bindings).
Proof.
  unfold order_as.
  induction pattern as [ | named pattern IH]; cbn [fold_right];
    [ apply Permutation_refl | ].
  eapply perm_trans; [ exact IH | apply to_front_permutation ].
Qed.

Fixpoint order_statics_as_scoped (e : expr) :
  ∀ pattern : expr, scoped e (order_statics_as pattern e).
Proof.
  intros pattern.
  destruct e as [x kind d body | bindings body | k h body | handlers body
                | ap | ac | scrutinee arms | ];
    try apply scoped_refl.
  - destruct pattern; cbn [order_statics_as]; try apply scoped_refl.
    apply scoped_let; [ apply incl_refl | apply order_statics_as_scoped ].
  - destruct pattern as [ | pattern_bindings pattern | | | | | | ];
      cbn [order_statics_as]; try apply scoped_refl.
    apply scoped_let_static; [ | apply order_statics_as_scoped ].
    apply scoped_bindings_permutation, order_as_permutation.
  - destruct h as [params metadata handler].
    destruct pattern as [ | | pattern_k pattern_h pattern | | | | | ];
      cbn [order_statics_as]; try apply scoped_refl.
    destruct pattern_h as [pattern_params pattern_metadata pattern_handler].
    apply scoped_let_cont; [ | apply order_statics_as_scoped ].
    apply scoped_handler_intro, order_statics_as_scoped.
  - destruct pattern; cbn [order_statics_as]; try apply scoped_refl.
    apply scoped_let_cont_recursive; [ reflexivity | | ].
    + apply scoped_handlers_refl.
    + apply order_statics_as_scoped.
Qed.

Theorem order_statics_as_keeps_scope : order_statics_as_keeps_scope_statement.
Proof.
  intros pattern. apply keeps_scope_intro. intros e.
  apply order_statics_as_scoped.
Qed.

(* ======================================================================== *)
(* The lifting                                                              *)
(* ======================================================================== *)

(* What a group of bindings that stands for a defining expression is
   asked: no code, no variable but those of the expression. *)
Definition stands_for (bindings : list static_binding) (d : named) : Prop :=
  forallb binding_code_closed bindings = true
  ∧ incl (flat_map free_variables_of_binding bindings)
      (free_variables_of_named d)
  ∧ flat_map free_continuations_of_binding bindings = [].

(* The bindings of the closures of a set: each binds a symbol to the
   set, with its value slots. *)
Lemma closure_bindings_stand_for (all l : list (function_slot * symbol))
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (slot : function_slot) :
  stands_for
    (map
       (fun '(slot, s) =>
          Bind_symbol s
            (Static_const_closure_of_set slot all function_decls value_slots))
       l)
    (Closure_of_set slot function_decls value_slots).
Proof.
  unfold stands_for.
  induction l as [ | [slot' s] l IH]; cbn [map forallb flat_map].
  - split; [ reflexivity | split; [ apply incl_nil_l | reflexivity ] ].
  - destruct IH as (Hclosed & Hv & Hk). scope.
    cbn [free_variables_of_static_const free_variables_of_named] in Hv |- *.
    rewrite Hclosed, Hk.
    split; [ reflexivity | split; [ | reflexivity ] ].
    apply incl_app; [ apply incl_refl | exact Hv ].
Qed.

Lemma lift_named_stands_for (target : lift_to) (d : named)
    (bindings : list static_binding) (s : symbol) :
  lift_named target d = Some (bindings, s) -> stands_for bindings d.
Proof.
  unfold lift_named. intros Hlift.
  repeat
    match type of Hlift with
    | match ?x with _ => _ end = Some _ =>
      destruct x eqn:?; try discriminate Hlift
    | (if ?c then _ else _) = Some _ =>
      destruct c eqn:?; try discriminate Hlift
    end;
    injection Hlift as <- <-;
    try apply closure_bindings_stand_for;
    unfold stands_for; cbn [forallb flat_map]; scope;
    cbn [free_variables_of_static_const free_variables_of_named
         free_variables_of_simples flat_map];
    rewrite ?app_nil_r;
    (split; [ reflexivity | split; [ apply incl_refl | reflexivity ] ]).
Qed.

(* A [Let] lifted: a group that stands for its defining expression, and
   a [Let] of a symbol. *)
Lemma scoped_let_lifted (x : variable) (kind : flambda_kind) (d : named)
    (bindings : list static_binding) (s : symbol) (body body' : expr) :
  stands_for bindings d ->
  scoped body body' ->
  scoped (Let x kind d body)
    (Let_static bindings (Let x kind (Simple (Symbol s)) body')).
Proof.
  intros (Hclosed_b & Hv_b & Hk_b) Hbody Hclosed. scope_in Hclosed. scope.
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  rewrite Hclosed_b, Hclosed', Hk_b.
  split; [ reflexivity | split; [ | exact Hk ] ].
  apply incl_app_app; [ exact Hv_b | ].
  cbn [free_variables_of_named free_variables_of_simple app].
  apply incl_without, Hv.
Qed.

Fixpoint lift_expr_scoped (lets : list (path * lift_to)) (e : expr)
    {struct e} :
  ∀ at_path : path, scoped e (lift_expr lets at_path e)
with lift_handler_scoped (lets : list (path * lift_to))
    (h : continuation_handler) {struct h} :
  ∀ at_path : path, scoped_handler h (lift_handler lets at_path h).
Proof.
  - intros at_path.
    destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [lift_expr]; try apply scoped_refl.
    + destruct (lookup at_path lets) as [target | ];
        [ | apply scoped_let; [ apply incl_refl | apply lift_expr_scoped ] ].
      destruct (lift_named target d) as [[bindings s] | ] eqn:Hlift;
        [ | apply scoped_let; [ apply incl_refl | apply lift_expr_scoped ] ].
      apply scoped_let_lifted;
        [ exact (lift_named_stands_for _ _ _ _ Hlift)
        | apply lift_expr_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply lift_expr_scoped ].
    + apply scoped_let_cont;
        [ apply lift_handler_scoped | apply lift_expr_scoped ].
    + apply scoped_let_cont_recursive; [ reflexivity | | ].
      * apply scoped_handlers_refl.
      * apply lift_expr_scoped.
  - intros at_path. destruct h as [params metadata handler].
    cbn [lift_handler]. apply scoped_handler_intro, lift_expr_scoped.
Qed.

Theorem lift_expr_keeps_scope : lift_expr_keeps_scope_statement.
Proof.
  intros lets at_path. apply keeps_scope_intro. intros e.
  apply lift_expr_scoped.
Qed.

Fixpoint bind_my_closures_scoped (known : list (code_id * symbol)) (e : expr)
    {struct e} :
  scoped e (bind_my_closures known e)
with bind_my_closures_in_handler_scoped (known : list (code_id * symbol))
    (h : continuation_handler) {struct h} :
  scoped_handler h (bind_my_closures_in_handler known h)
with bind_my_closure_scoped (known : list (code_id * symbol)) (id : code_id)
    (c : code) {struct c} :
  scoped_code c (bind_my_closure known id c).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [bind_my_closures]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply bind_my_closures_scoped ].
    + apply scoped_let_static; [ | apply bind_my_closures_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ apply scoped_bindings_nil | ].
      destruct binding as [id c | id | sym const];
        cbn [bind_my_closures_in_binding].
      * apply scoped_bindings_cons_code;
          [ apply bind_my_closure_scoped | exact IH ].
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_same, IH.
    + apply scoped_let_cont;
        [ apply bind_my_closures_in_handler_scoped
        | apply bind_my_closures_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply bind_my_closures_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply bind_my_closures_in_handler_scoped | exact IH ].
  - destruct h as [params metadata handler].
    cbn [bind_my_closures_in_handler].
    apply scoped_handler_intro, bind_my_closures_scoped.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [bind_my_closure].
    destruct (lookup id known) as [s | ];
      [ | apply scoped_code_intro, bind_my_closures_scoped ].
    destruct (among my_closure params);
      [ apply scoped_code_intro, bind_my_closures_scoped | ].
    apply scoped_code_intro.
    eapply scoped_trans;
      [ apply bind_my_closures_scoped | apply scoped_let_closed; reflexivity ].
Qed.

Theorem bind_my_closures_keeps_scope : bind_my_closures_keeps_scope_statement.
Proof.
  intros known. apply keeps_scope_intro. intros e.
  apply bind_my_closures_scoped.
Qed.

(* The lifting out of code. A defining expression that mentions no
   variable has none free, so the bindings that stand for it have
   none. *)
Lemma plain_simples_free (args : list simple) :
  forallb plain_simple args = true -> free_variables_of_simples args = [].
Proof.
  induction args as [ | s args IH]; cbn [forallb]; [ reflexivity | ].
  intros Hplain. apply andb_true_iff in Hplain. destruct Hplain as [Hs Hargs].
  destruct s as [x | sym | c]; [ discriminate Hs | | ];
    cbn [free_variables_of_simples flat_map free_variables_of_simple app];
    exact (IH Hargs).
Qed.

Lemma plain_slots_free (slots : list (value_slot * simple)) :
  forallb (fun slot => plain_simple (snd slot)) slots = true ->
  free_variables_of_value_slots slots = [].
Proof.
  induction slots as [ | [slot s] slots IH]; cbn [forallb];
    [ reflexivity | ].
  intros Hplain. apply andb_true_iff in Hplain. destruct Hplain as [Hs Hslots].
  destruct s as [x | sym | c]; [ discriminate Hs | | ];
    cbn [free_variables_of_value_slots flat_map free_variables_of_simple
         snd app];
    exact (IH Hslots).
Qed.

Lemma closed_named_free (d : named) :
  closed_named d = true -> free_variables_of_named d = [].
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    cbn [closed_named free_variables_of_named]; try discriminate;
    intros Hclosed.
  - exact (plain_simples_free args Hclosed).
  - exact (plain_slots_free slots Hclosed).
  - exact (plain_slots_free slots Hclosed).
Qed.

Definition no_free_names (bindings : list static_binding) : Prop :=
  forallb binding_code_closed bindings = true
  ∧ flat_map free_variables_of_binding bindings = []
  ∧ flat_map free_continuations_of_binding bindings = [].

Lemma no_free_names_app (bindings bindings' : list static_binding) :
  no_free_names bindings -> no_free_names bindings' ->
  no_free_names (bindings ++ bindings').
Proof.
  intros (Hclosed & Hv & Hk) (Hclosed' & Hv' & Hk'). unfold no_free_names.
  rewrite forallb_app, !flat_map_app, Hclosed, Hclosed', Hv, Hv', Hk, Hk'.
  repeat split.
Qed.

Lemma lifted_out_no_free_names (lets : list (path * lift_to))
    (at_path : path) (d : named) :
  no_free_names
    match lifted_out lets at_path d with
    | Some (bindings, _) => bindings
    | None => []
    end.
Proof.
  unfold lifted_out.
  destruct (lookup at_path lets) as [target | ]; [ | repeat split ].
  destruct (closed_named d) eqn:Hclosed; [ | repeat split ].
  destruct (lift_named target d) as [[bindings s] | ] eqn:Hlift;
    [ | repeat split ].
  destruct (lift_named_stands_for target d bindings s Hlift)
    as (Hcode & Hv & Hk).
  rewrite (closed_named_free d Hclosed) in Hv.
  split; [ exact Hcode | split; [ | exact Hk ] ].
  exact (incl_l_nil Hv).
Qed.

Fixpoint lifted_out_bindings_no_free_names (lets : list (path * lift_to))
    (at_path : path) (e : expr) {struct e} :
  no_free_names (lifted_out_bindings lets at_path e)
with handler_lifted_out_bindings_no_free_names
    (lets : list (path * lift_to)) (at_path : path)
    (h : continuation_handler) {struct h} :
  no_free_names (handler_lifted_out_bindings lets at_path h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [lifted_out_bindings]; try solve [ repeat split ].
    + apply no_free_names_app;
        [ apply lifted_out_no_free_names
        | apply lifted_out_bindings_no_free_names ].
    + apply lifted_out_bindings_no_free_names.
    + apply no_free_names_app;
        [ apply lifted_out_bindings_no_free_names
        | apply handler_lifted_out_bindings_no_free_names ].
    + apply no_free_names_app; [ apply lifted_out_bindings_no_free_names | ].
      generalize (xI at_path).
      induction handlers as [ | [k1 h1] handlers IH]; intros q;
        cbn [rec_handlers_bindings]; [ repeat split | ].
      apply no_free_names_app;
        [ apply handler_lifted_out_bindings_no_free_names | apply IH ].
  - destruct h as [params metadata handler].
    cbn [handler_lifted_out_bindings].
    apply lifted_out_bindings_no_free_names.
Qed.

(* Bindings with no free names, put once each, have none. *)
Lemma once_each_no_free_names :
  ∀ bindings seen : list static_binding,
    no_free_names bindings -> no_free_names (once_each seen bindings).
Proof.
  induction bindings as [ | binding bindings IH]; intros seen;
    cbn [once_each]; [ exact (fun H => H) | ].
  intros (Hclosed & Hv & Hk). cbn [forallb flat_map] in Hclosed, Hv, Hk.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  apply app_eq_nil in Hv. destruct Hv as [Hv Hv'].
  apply app_eq_nil in Hk. destruct Hk as [Hk Hk'].
  assert (Hbindings : no_free_names bindings) by (repeat split; assumption).
  destruct (existsb (same_binding binding) seen);
    [ exact (IH _ Hbindings) | ].
  destruct (IH (binding :: seen) Hbindings) as (Hclosed1 & Hv1 & Hk1).
  unfold no_free_names. cbn [forallb flat_map].
  rewrite Hfirst, Hclosed1, Hv, Hv1, Hk, Hk1. repeat split.
Qed.

(* Bindings with no free names put before a group. *)
Lemma scoped_bindings_prepend (first bindings bindings' : list static_binding) :
  no_free_names first ->
  scoped_bindings bindings bindings' ->
  scoped_bindings bindings (first ++ bindings').
Proof.
  intros (Hcode & Hv & Hk) Hbindings Hclosed.
  destruct (Hbindings Hclosed) as (Hclosed' & Hv' & Hk').
  rewrite forallb_app, !flat_map_app, Hcode, Hclosed', Hv, Hk.
  split; [ reflexivity | split; assumption ].
Qed.

Lemma scoped_bindings_app (bindings1 bindings1' bindings2 bindings2'
    : list static_binding) :
  scoped_bindings bindings1 bindings1' ->
  scoped_bindings bindings2 bindings2' ->
  scoped_bindings (bindings1 ++ bindings2) (bindings1' ++ bindings2').
Proof.
  intros H1 H2 Hclosed. rewrite forallb_app in Hclosed.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hclosed1 Hclosed2].
  destruct (H1 Hclosed1) as (Hclosed1' & Hv1 & Hk1).
  destruct (H2 Hclosed2) as (Hclosed2' & Hv2 & Hk2).
  rewrite forallb_app, !flat_map_app, Hclosed1', Hclosed2'.
  split; [ reflexivity | split; apply incl_app_app; assumption ].
Qed.

Fixpoint lift_from_code_scoped (K : lets_in_code)
    (lets : list (path * lift_to)) (at_path : path) (e : expr) {struct e} :
  scoped e (lift_from_code K lets at_path e)
with handler_lift_from_code_scoped (K : lets_in_code)
    (lets : list (path * lift_to)) (at_path : path)
    (h : continuation_handler) {struct h} :
  scoped_handler h (handler_lift_from_code K lets at_path h)
with binding_lift_from_code_scoped (K : lets_in_code)
    (binding : static_binding) {struct binding} :
  scoped_bindings [binding] (binding_lift_from_code K binding).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [lift_from_code]; try apply scoped_refl.
    + destruct (lifted_out lets at_path d) as [[bindings s] | ];
        (apply scoped_let; [ | apply lift_from_code_scoped ]);
        [ apply incl_nil_l | apply incl_refl ].
    + apply scoped_let_static; [ | apply lift_from_code_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [flat_map];
        [ apply scoped_bindings_nil | ].
      exact (scoped_bindings_app [binding] _ bindings _
               (binding_lift_from_code_scoped K binding) IH).
    + apply scoped_let_cont;
        [ apply handler_lift_from_code_scoped | apply lift_from_code_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply lift_from_code_scoped ].
      * generalize (xI at_path).
        induction handlers as [ | [k1 h1] handlers IH]; intros q;
          [ reflexivity | ].
        cbn [rec_handlers_map map fst]. rewrite IH. reflexivity.
      * generalize (xI at_path).
        induction handlers as [ | [k1 h1] handlers IH]; intros q;
          cbn [rec_handlers_map]; [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply handler_lift_from_code_scoped | apply IH ].
  - destruct h as [params metadata handler].
    cbn [handler_lift_from_code].
    apply scoped_handler_intro, lift_from_code_scoped.
  - destruct binding as [id c | id | sym const];
      cbn [binding_lift_from_code]; try apply scoped_bindings_refl.
    destruct c as [params my_closure ret exn metadata body].
    apply scoped_bindings_prepend;
      [ apply once_each_no_free_names, lifted_out_bindings_no_free_names | ].
    apply scoped_bindings_cons_code;
      [ apply scoped_code_intro, lift_from_code_scoped
      | apply scoped_bindings_nil ].
Qed.

Theorem lift_from_code_keeps_scope : lift_from_code_keeps_scope_statement.
Proof.
  intros K lets at_path. apply keeps_scope_intro. intros e.
  apply lift_from_code_scoped.
Qed.

(* The lifting to the head of the unit: the unit with symbols in the
   place of defining expressions that have no variable, under a group
   of bindings that have no free name. *)
Lemma scoped_let_static_around (bindings : list static_binding)
    (e e' : expr) :
  no_free_names bindings -> scoped e e' -> scoped e (Let_static bindings e').
Proof.
  intros (Hcode & Hv & Hk) He Hclosed.
  destruct (He Hclosed) as (Hclosed' & Hv' & Hk'). scope.
  rewrite Hcode, Hclosed', Hv, Hk.
  split; [ reflexivity | split; assumption ].
Qed.

Lemma lift_to_head_scoped (lets : list (path * lift_to)) (e : expr) :
  scoped e (lift_to_head lets e).
Proof.
  unfold lift_to_head. cbv zeta.
  pose proof
    (once_each_no_free_names _ []
       (lifted_out_bindings_no_free_names lets root e))
    as Hfree.
  destruct (once_each [] (lifted_out_bindings lets root e))
    as [ | binding bindings];
    [ apply scoped_refl | ].
  exact (scoped_let_static_around _ _ _ Hfree
           (lift_from_code_scoped [] lets root e)).
Qed.

Theorem lift_to_head_keeps_scope : lift_to_head_keeps_scope_statement.
Proof.
  intros lets. apply keeps_scope_intro. intros e. apply lift_to_head_scoped.
Qed.

(* Each of the three liftings as the check applies it: made, or the
   term as it is. *)
Lemma lift_to_head_unit_scoped (u : unit_interface)
    (lets : list (path * lift_to)) (e : expr) :
  scoped e
    match lift_to_head_unit u lets e with
    | Some e' => e'
    | None => e
    end.
Proof.
  unfold lift_to_head_unit. cbv zeta.
  match goal with
  | |- scoped _ (match (if ?c then _ else _) with _ => _ end) => destruct c
  end;
    [ apply lift_to_head_scoped | apply scoped_refl ].
Qed.

Lemma lift_from_code_unit_scoped (u : unit_interface) (K : lets_in_code)
    (e : expr) :
  scoped e
    match lift_from_code_unit u K e with
    | Some e' => e'
    | None => e
    end.
Proof.
  unfold lift_from_code_unit. cbv zeta.
  match goal with
  | |- scoped _ (match (if ?c then _ else _) with _ => _ end) => destruct c
  end;
    [ apply lift_from_code_scoped | apply scoped_refl ].
Qed.

Lemma lift_unit_scoped (L : lifting) (e : expr) :
  scoped e
    match lift_unit L e with
    | Some e' => e'
    | None => e
    end.
Proof.
  unfold lift_unit. cbv zeta.
  match goal with
  | |- scoped _ (match (if ?c then _ else _) with _ => _ end) => destruct c
  end;
    [ | apply scoped_refl ].
  eapply scoped_trans;
    [ apply lift_expr_scoped | apply bind_my_closures_scoped ].
Qed.

(* ======================================================================== *)
(* The lifting to a group around                                            *)
(* ======================================================================== *)

(* A group that may be put binds symbols only: it has no code and
   mentions no continuation. *)
Lemma putable_group_closed (bindings : list static_binding) :
  putable_group bindings = true ->
  forallb binding_code_closed bindings = true
  ∧ flat_map free_continuations_of_binding bindings = [].
Proof.
  unfold putable_group.
  induction bindings as [ | binding bindings IH]; cbn [forallb flat_map];
    [ intros _; split; reflexivity | ].
  intros H. apply andb_true_iff in H. destruct H as [Hfirst Hrest].
  destruct (IH Hrest) as [Hclosed Hk].
  destruct binding as [id c | id | s const]; try discriminate.
  scope. rewrite Hclosed, Hk. split; reflexivity.
Qed.

(* A group put before a term of which it mentions only variables. *)
Lemma scoped_group_put (bindings : list static_binding) (e e' : expr) :
  putable_group bindings = true ->
  group_in_scope bindings e = true ->
  scoped e e' ->
  scoped e
    match bindings with
    | [] => e'
    | binding :: bindings' => Let_static (binding :: bindings') e'
    end.
Proof.
  intros Hput Hscope He. unfold group_in_scope in Hscope.
  destruct bindings as [ | binding bindings']; [ exact He | ].
  set (bindings := binding :: bindings') in *.
  intros Hclosed. destruct (He Hclosed) as (Hclosed' & Hv & Hk).
  destruct (putable_group_closed bindings Hput) as [Hb Hbk].
  scope. rewrite Hb, Hclosed', Hbk. split; [ reflexivity | ]. cbn [app].
  split; [ | exact Hk ].
  apply incl_app; [ | exact Hv ].
  intros x Hx. apply among_In.
  exact
    (proj1 (FlambdaMapProofs.all_among _ _) Hscope x
       (proj2 (among_In _ _) Hx)).
Qed.

Fixpoint lift_around_scoped (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (e : expr) {struct e} :
  around_ok lets groups known at_path e = true ->
  scoped e (lift_around lets groups at_path e)
with handler_lift_around_scoped (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (h : continuation_handler) {struct h} :
  handler_around_ok lets groups known at_path h = true ->
  scoped_handler h (lift_around_handler lets groups at_path h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [around_ok lift_around]; intros Hok; try apply scoped_refl.
    + apply andb_true_iff in Hok. destruct Hok as [_ Hbody].
      destruct (alias_of lets at_path d) as [[bindings s] | ];
        (apply scoped_let;
         [ | exact (lift_around_scoped _ _ _ _ _ Hbody) ]);
        [ intros x' Hx'; cbn in Hx'; destruct Hx' | apply incl_refl ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl
        | exact (lift_around_scoped _ _ _ _ _ Hok) ].
    + cbv zeta in Hok.
      apply andb_true_iff in Hok. destruct Hok as [Hok Hbody].
      apply andb_true_iff in Hok. destruct Hok as [Hok Hh].
      apply andb_true_iff in Hok. destruct Hok as [Hput Hscope].
      apply (scoped_group_put _ _ _ Hput Hscope).
      apply scoped_let_cont;
        [ exact (handler_lift_around_scoped _ _ _ _ _ Hh)
        | exact (lift_around_scoped _ _ _ _ _ Hbody) ].
    + cbv zeta in Hok.
      apply andb_true_iff in Hok. destruct Hok as [Hok Hbody].
      apply andb_true_iff in Hok. destruct Hok as [Hok Hhandlers].
      apply andb_true_iff in Hok. destruct Hok as [Hok _].
      apply andb_true_iff in Hok. destruct Hok as [Hput Hscope].
      apply (scoped_group_put _ _ _ Hput Hscope). clear Hscope.
      apply scoped_let_cont_recursive;
        [ | | exact (lift_around_scoped _ _ _ _ _ Hbody) ].
      * clear Hhandlers. generalize (xI at_path).
        induction handlers as [ | [k1 h1] handlers IH]; intros q;
          [ reflexivity | ].
        cbn [rec_handlers_map map fst]. rewrite IH. reflexivity.
      * revert Hhandlers. generalize (xI at_path).
        induction handlers as [ | [k1 h1] handlers IH]; intros q Hhandlers;
          cbn [rec_handlers_map]; [ apply scoped_handlers_nil | ].
        cbn [rec_handlers_forall] in Hhandlers.
        apply andb_true_iff in Hhandlers. destruct Hhandlers as [H1 Hrest].
        apply scoped_handlers_cons;
          [ exact (handler_lift_around_scoped _ _ _ _ _ H1)
          | exact (IH _ Hrest) ].
  - destruct h as [params metadata handler].
    cbn [handler_around_ok lift_around_handler]. intros Hok.
    apply scoped_handler_intro.
    exact (lift_around_scoped _ _ _ _ _ Hok).
Qed.

Theorem lift_around_keeps_scope : lift_around_keeps_scope_statement.
Proof.
  intros lets groups at_path. apply keeps_scope_intro. intros e.
  destruct (around_ok lets groups [] at_path e) eqn:Hok;
    [ exact (lift_around_scoped _ _ _ _ _ Hok) | apply scoped_refl ].
Qed.

Lemma lift_around_unit_scoped (u : unit_interface)
    (lets : list (path * lift_to)) (places : list (path * list path))
    (e : expr) :
  scoped e
    match lift_around_unit u lets places e with
    | Some e' => e'
    | None => e
    end.
Proof.
  unfold lift_around_unit. cbv zeta.
  set (groups := groups_of lets places e).
  destruct (exits_plain_knowing [] [] e); cbn [andb];
    [ | apply scoped_refl ].
  destruct (FlambdaHeapFreeSpec.expr_free_of_heap e); cbn [andb];
    [ | apply scoped_refl ].
  destruct (around_ok lets groups [] root e) eqn:Hok; cbn [andb];
    [ | apply scoped_refl ].
  destruct (well_formed_unit u (lift_around lets groups root e));
    [ exact (lift_around_scoped _ _ _ _ _ Hok) | apply scoped_refl ].
Qed.

(* The tables bound at the head of a unit
   (FlambdaUnusedHeadStaticsSpec.add_head_tables): a table whose
   elements are constants and symbols has no code and mentions no
   variable. *)
Lemma plain_simples_closed (elements : list simple) :
  forallb plain_simple elements = true ->
  free_variables_of_simples elements = [].
Proof.
  induction elements as [ | s elements IH]; [ reflexivity | ].
  cbn [forallb]. intros Hplain.
  apply andb_true_iff in Hplain. destruct Hplain as [Hs Hplain].
  unfold free_variables_of_simples in *. cbn [flat_map].
  rewrite (IH Hplain). destruct s; [ discriminate | reflexivity | reflexivity ].
Qed.

Lemma tables_closed (tables : list (symbol * list simple)) :
  tables_plain tables = true ->
  forallb binding_code_closed (map table_binding tables) = true
  ∧ flat_map free_variables_of_binding (map table_binding tables) = []
  ∧ flat_map free_continuations_of_binding (map table_binding tables) = [].
Proof.
  unfold tables_plain.
  induction tables as [ | [s elements] tables IH]; [ repeat split | ].
  cbn [forallb snd]. intros Hplain.
  apply andb_true_iff in Hplain. destruct Hplain as [Helements Hplain].
  destruct (IH Hplain) as (Hcode & Hv & Hk).
  cbn [map forallb flat_map].
  change (table_binding (s, elements))
    with (Bind_symbol s (Static_const_immutable_value_array elements)).
  scope. cbn [free_variables_of_static_const].
  rewrite Hcode, Hv, Hk, (plain_simples_closed elements Helements).
  repeat split.
Qed.

Lemma add_head_tables_scoped (tables : list (symbol * list simple))
    (e : expr) :
  scoped e (add_head_tables tables e).
Proof.
  unfold add_head_tables.
  destruct tables as [ | table tables]; [ apply scoped_refl | ].
  cbv zeta.
  destruct
    (tables_plain (table :: tables)
     && head_statics_removable (tables_dropped (table :: tables))
          (Let_static (map table_binding (table :: tables)) e))
    eqn:Hchecks;
    [ | apply scoped_refl ].
  apply andb_true_iff in Hchecks. destruct Hchecks as [Hplain _].
  destruct (tables_closed _ Hplain) as (Hcode & Hv & Hk).
  intros Hclosed. scope. rewrite Hcode, Hv, Hk. cbn [andb app].
  split; [ exact Hclosed | split; apply incl_refl ].
Qed.

(* The copies of code for static sets of closures
   (FlambdaCopyCodeSpec.with_copies): the code bound a second time is
   closed if the code is, a static set has the value slots that it
   had, and a call has its callee and its arguments. *)
Lemma scoped_bindings_cons_code_twice (id id' id'' : code_id) (c c' : code)
    (bindings bindings' : list static_binding) :
  scoped_code c c' ->
  scoped_bindings bindings bindings' ->
  scoped_bindings (Bind_code id c :: bindings)
    (Bind_code id' c' :: Bind_code id'' c' :: bindings').
Proof.
  intros Hc Hbindings Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hbindings Hrest) as (Hrest' & Hv & Hk).
  destruct (code_is_closed_free c' (Hc Hfirst)) as [Hv' Hk'].
  rewrite (Hc Hfirst), Hrest', Hv', Hk'.
  split; [ reflexivity | split; apply incl_appr; assumption ].
Qed.

Lemma copy_call_scoped (ids : list (code_id * code_id))
    (symbols : list symbol) (ap : apply_expr) :
  scoped (Apply ap) (Apply (copy_call ids symbols ap)).
Proof.
  destruct ap as [callee kind args k exn extra args_arity return_arity].
  unfold copy_call. cbn [ap_call_kind ap_callee].
  destruct kind as [id | ]; [ | apply scoped_refl ].
  destruct callee as [x | s | c];
    [ destruct (lookup id ids) | destruct (among s symbols)
    | destruct (lookup id ids) ];
    try apply scoped_refl;
    intros Hclosed; split; try exact Hclosed; split; apply incl_refl.
Qed.

Fixpoint copy_expr_scoped (ids : list (code_id * code_id))
    (symbols : list symbol) (e : expr) {struct e} :
  scoped e (copy_expr ids symbols e)
with copy_handler_scoped (ids : list (code_id * code_id))
    (symbols : list symbol) (h : continuation_handler) {struct h} :
  scoped_handler h (copy_handler ids symbols h)
with copy_code_scoped (ids : list (code_id * code_id))
    (symbols : list symbol) (c : code) {struct c} :
  scoped_code c (copy_code ids symbols c).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [copy_expr]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply copy_expr_scoped ].
    + apply scoped_let_static; [ | apply copy_expr_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [flat_map];
        [ apply scoped_bindings_nil | ].
      destruct binding as [id c | id | sym const]; cbn [copy_binding app].
      * destruct (lookup id ids) as [id' | ]; cbn [app];
          [ apply scoped_bindings_cons_code_twice
          | apply scoped_bindings_cons_code ];
          solve [ apply copy_code_scoped | exact IH ].
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_symbol; [ | exact IH ].
        destruct (among sym symbols); [ | apply incl_refl ].
        destruct const; apply incl_refl.
    + apply scoped_let_cont;
        [ apply copy_handler_scoped | apply copy_expr_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply copy_expr_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply copy_handler_scoped | exact IH ].
    + apply copy_call_scoped.
  - destruct h as [params metadata handler]. cbn [copy_handler].
    apply scoped_handler_intro, copy_expr_scoped.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [copy_code]. apply scoped_code_intro, copy_expr_scoped.
Qed.

Lemma with_copies_scoped
    (copies : list (list (code_id * code_id) * list symbol)) :
  ∀ e : expr, scoped e (with_copies copies e).
Proof.
  induction copies as [ | [ids symbols] copies IH]; intros e;
    cbn [with_copies]; [ apply scoped_refl | ].
  unfold copy_static_code. destruct (copy_ok ids e); [ | apply IH ].
  exact (scoped_trans _ _ _ (copy_expr_scoped ids symbols e) (IH _)).
Qed.

Theorem with_lifting_keeps_scope : with_lifting_keeps_scope_statement.
Proof.
  intros L. apply keeps_scope_intro. intros e.
  unfold with_lifting. cbv zeta.
  eapply scoped_trans; [ | apply with_copies_scoped ].
  eapply scoped_trans; [ | apply add_head_tables_scoped ].
  eapply scoped_trans; [ apply lift_to_head_unit_scoped | ].
  eapply scoped_trans; [ apply lift_from_code_unit_scoped | ].
  eapply scoped_trans;
    [ apply lift_unit_scoped | apply lift_around_unit_scoped ].
Qed.

(* The four liftings without the two tests of the unit
   (FlambdaLiftObservedSpec.v), as the four with them. *)
Lemma to_head_observed_scoped (L : lifting) (e : expr) :
  scoped e (to_head_observed L e).
Proof.
  unfold to_head_observed, lift_to_head_unit_observed. cbv zeta.
  match goal with
  | |- scoped _ (match (if ?c then _ else _) with _ => _ end) => destruct c
  end;
    [ apply lift_to_head_scoped | apply scoped_refl ].
Qed.

Lemma out_of_code_observed_scoped (L : lifting) (e : expr) :
  scoped e (out_of_code_observed L e).
Proof.
  unfold out_of_code_observed, lift_from_code_unit_observed. cbv zeta.
  match goal with
  | |- scoped _ (match (if ?c then _ else _) with _ => _ end) => destruct c
  end;
    [ apply lift_from_code_scoped | apply scoped_refl ].
Qed.

Lemma lets_observed_scoped (L : lifting) (e : expr) :
  scoped e (lets_observed L e).
Proof.
  unfold lets_observed, lift_unit_observed. cbv zeta.
  match goal with
  | |- scoped _ (match (if ?c then _ else _) with _ => _ end) => destruct c
  end;
    [ | apply scoped_refl ].
  eapply scoped_trans;
    [ apply lift_expr_scoped | apply bind_my_closures_scoped ].
Qed.

Lemma around_observed_scoped (L : lifting) (e : expr) :
  scoped e (around_observed L e).
Proof.
  unfold around_observed, lift_around_unit_observed. cbv zeta.
  set (groups := groups_of (lifted_around L) (groups_put L) e).
  destruct (around_ok (lifted_around L) groups [] root e) eqn:Hok;
    cbn [andb]; [ | apply scoped_refl ].
  destruct
    (well_formed_unit (lifting_unit L)
       (lift_around (lifted_around L) groups root e));
    [ exact (lift_around_scoped _ _ _ _ _ Hok) | apply scoped_refl ].
Qed.

Theorem with_lifting_observed_keeps_scope :
  with_lifting_observed_keeps_scope_statement.
Proof.
  intros L. apply keeps_scope_intro. intros e.
  unfold with_lifting_observed, liftings_observed.
  eapply scoped_trans; [ | apply with_copies_scoped ].
  eapply scoped_trans; [ | apply add_head_tables_scoped ].
  eapply scoped_trans; [ apply to_head_observed_scoped | ].
  eapply scoped_trans; [ apply out_of_code_observed_scoped | ].
  eapply scoped_trans;
    [ apply lets_observed_scoped | apply around_observed_scoped ].
Qed.

(* ======================================================================== *)
(* What the unit binds its symbols to                                       *)
(* ======================================================================== *)

Lemma fold_known_named_free (statics : static_env) (d : named) :
  incl (free_variables_of_named (fold_known_named statics d))
    (free_variables_of_named d).
Proof.
  unfold fold_known_named.
  destruct (named_known statics d); [ | apply incl_refl ].
  destruct (eval_named statics [] d) as [v | ]; [ | apply incl_refl ].
  destruct v; try apply incl_refl; apply incl_nil_l.
Qed.

Lemma fold_function_slot_free (sets : static_env) (d : named) :
  incl (free_variables_of_named (fold_function_slot sets d))
    (free_variables_of_named d).
Proof.
  unfold fold_function_slot.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | pfrom slot closure | move_from move_to closure];
    try apply incl_refl.
  destruct closure as [x | sym | c]; try apply incl_refl.
  match goal with
  | |- context [eval_named ?sets ?vars ?d] =>
    destruct (eval_named sets vars d) as [v | ]
  end;
    [ | apply incl_refl ].
  destruct v; try apply incl_refl; apply incl_nil_l.
Qed.

Lemma fold_block_load_free (sets : static_env) (d : named) :
  incl (free_variables_of_named (fold_block_load sets d))
    (free_variables_of_named d).
Proof.
  unfold fold_block_load.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | pfrom slot closure | move_from move_to closure];
    try apply incl_refl.
  destruct p as [p | p | p | p | rp]; try apply incl_refl.
  destruct p; try apply incl_refl.
  destruct mut; try apply incl_refl;
    (destruct args as [ | [x | sym | c] [ | arg args]]; try apply incl_refl;
     match goal with
     | |- context [eval_named ?sets ?vars ?d] =>
       destruct (eval_named sets vars d) as [v | ]
     end;
     [ | apply incl_refl ];
     destruct v; try apply incl_refl; apply incl_nil_l).
Qed.

Lemma fold_known_apply_scoped (sets : static_env) (ap : apply_expr) :
  scoped (Apply ap) (Apply (fold_known_apply sets ap)).
Proof.
  destruct ap as [callee call_kind args k exn extra args_arity return_arity].
  unfold fold_known_apply. cbn [ap_call_kind ap_callee].
  destruct call_kind as [code | ]; [ apply scoped_refl | ].
  destruct callee as [x | s | c]; try apply scoped_refl.
  destruct (closure_of sets (Value_symbol s)) as [[code slots] | ];
    [ | apply scoped_refl ].
  intros Hclosed. split; [ exact Hclosed | split; apply incl_refl ].
Qed.

Fixpoint fold_known_scoped (statics sets : static_env) (e : expr)
    {struct e} :
  scoped e (fold_known statics sets e)
with fold_known_handler_scoped (statics sets : static_env)
    (h : continuation_handler) {struct h} :
  scoped_handler h (fold_known_handler statics sets h)
with fold_known_code_scoped (statics sets : static_env) (c : code)
    {struct c} :
  scoped_code c (fold_known_code statics sets c).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [fold_known]; try apply scoped_refl.
    + apply scoped_let; [ | apply fold_known_scoped ].
      exact (incl_tran (fold_block_load_free sets _)
               (incl_tran (fold_function_slot_free sets _)
                  (fold_known_named_free statics d))).
    + apply scoped_let_static; [ | apply fold_known_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ apply scoped_bindings_nil | ].
      destruct binding as [id c | id | sym const]; cbn [fold_known_binding].
      * apply scoped_bindings_cons_code;
          [ apply fold_known_code_scoped | exact IH ].
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_same, IH.
    + apply scoped_let_cont;
        [ apply fold_known_handler_scoped | apply fold_known_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply fold_known_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply fold_known_handler_scoped | exact IH ].
    + apply fold_known_apply_scoped.
  - destruct h as [params metadata handler]. cbn [fold_known_handler].
    apply scoped_handler_intro, fold_known_scoped.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [fold_known_code]. apply scoped_code_intro, fold_known_scoped.
Qed.

Theorem fold_known_keeps_scope : fold_known_keeps_scope_statement.
Proof.
  intros statics sets. apply keeps_scope_intro. intros e.
  apply fold_known_scoped.
Qed.

Theorem with_known_keeps_scope : with_known_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e. unfold with_known, fold_known_unit.
  cbv zeta. destruct (bound_as (known_bindings e) (known_sets e) e);
    [ apply fold_known_scoped | apply scoped_refl ].
Qed.

(* ======================================================================== *)
(* The value slots that the unit gives one symbol or constant               *)
(* ======================================================================== *)

(* What is put in the place of a projection has no variable. *)
Lemma read_slot_free (known : known_slots) (syms : list symbol) (d : named) :
  closed_slots known = true ->
  incl (free_variables_of_named (read_slot known syms d))
    (free_variables_of_named d).
Proof.
  intros Hclosed. unfold closed_slots in Hclosed.
  rewrite forallb_forall in Hclosed.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | pfrom slot closure | move_from move_to closure];
    cbn [read_slot]; try apply incl_refl;
    (destruct (lookup slot known) as [a | ] eqn:Hknown;
       [ | apply incl_refl ]);
    (destruct (simple_statics_in_scope syms a); [ | apply incl_refl ]);
    specialize (Hclosed (slot, a) (lookup_in _ _ _ Hknown));
    (destruct a as [x | sym | k]; [ discriminate | | ]); apply incl_nil_l.
Qed.

Fixpoint read_slots_scoped (known : known_slots)
    (Hclosed : closed_slots known = true) (e : expr) {struct e} :
  ∀ syms : list symbol, scoped e (read_slots known syms e)
with read_slots_in_handler_scoped (known : known_slots)
    (Hclosed : closed_slots known = true) (h : continuation_handler)
    {struct h} :
  ∀ syms : list symbol,
    scoped_handler h (read_slots_in_handler known syms h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros syms; cbn [read_slots]; try apply scoped_refl.
    + apply scoped_let;
        [ apply read_slot_free, Hclosed | apply read_slots_scoped, Hclosed ].
    + apply scoped_let_static; [ | apply read_slots_scoped, Hclosed ].
      match goal with
      | |- scoped_bindings _ (map (read_slots_in_binding known ?all) _) =>
        generalize all
      end.
      intros all.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map]; [ apply scoped_bindings_nil | | | ].
      * cbn [read_slots_in_binding code_is_stub].
        destruct (cm_is_stub metadata);
          [ apply scoped_bindings_cons_same, IH | ].
        cbn [read_slots_in_code].
        apply scoped_bindings_cons_code; [ | exact IH ].
        apply scoped_code_intro, read_slots_scoped, Hclosed.
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_same, IH.
    + apply scoped_let_cont;
        [ apply read_slots_in_handler_scoped, Hclosed
        | apply read_slots_scoped, Hclosed ].
    + apply scoped_let_cont_recursive;
        [ | | apply read_slots_scoped, Hclosed ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * match goal with
        | |- scoped_handlers _
               (map (fun '(k, h) => (k, read_slots_in_handler known ?all h))
                  _) =>
          generalize all
        end.
        intros all.
        induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply read_slots_in_handler_scoped, Hclosed | exact IH ].
  - destruct h as [params metadata handler]. intros syms.
    cbn [read_slots_in_handler].
    apply scoped_handler_intro, read_slots_scoped, Hclosed.
Qed.

Theorem read_slots_keeps_scope : read_slots_keeps_scope_statement.
Proof.
  intros known syms Hclosed. apply keeps_scope_intro. intros e.
  apply read_slots_scoped, Hclosed.
Qed.

Theorem read_known_slots_keeps_scope : read_known_slots_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e. unfold read_known_slots. cbv zeta.
  destruct (closed_slots (slots_known (slots_made e))) eqn:Hclosed;
    cbn [andb]; [ | apply scoped_refl ].
  destruct (slots_as (slots_known (slots_made e)) e);
    [ apply read_slots_scoped, Hclosed | apply scoped_refl ].
Qed.

Theorem read_known_slots_anywhere_keeps_scope :
  read_known_slots_anywhere_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e. unfold read_known_slots_anywhere.
  cbv zeta.
  destruct (closed_slots (slots_known (slots_made e))) eqn:Hclosed;
    cbn [andb]; [ | apply scoped_refl ].
  destruct (slots_as (slots_known (slots_made e)) e);
    [ apply read_slots_scoped, Hclosed | apply scoped_refl ].
Qed.

(* ======================================================================== *)
(* The direct calls of the code of a stub                                   *)
(* ======================================================================== *)

Lemma demote_call_scoped (ids : list code_id) (ap : apply_expr) :
  scoped (Apply ap) (Apply (demote_call ids ap)).
Proof.
  destruct ap as [callee call_kind args k exn extra args_arity return_arity].
  unfold demote_call. cbn [ap_call_kind].
  destruct call_kind as [code | ]; [ | apply scoped_refl ].
  destruct (among code ids); [ | apply scoped_refl ].
  intros Hclosed. split; [ exact Hclosed | split; apply incl_refl ].
Qed.

Fixpoint demote_calls_scoped (ids : list code_id) (e : expr) {struct e} :
  scoped e (demote_calls ids e)
with demote_calls_in_handler_scoped (ids : list code_id)
    (h : continuation_handler) {struct h} :
  scoped_handler h (demote_calls_in_handler ids h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [demote_calls]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply demote_calls_scoped ].
    + apply scoped_let_static; [ | apply demote_calls_scoped ].
      induction bindings as [ | binding bindings IH];
        [ apply scoped_bindings_nil | apply scoped_bindings_cons_same, IH ].
    + apply scoped_let_cont;
        [ apply demote_calls_in_handler_scoped | apply demote_calls_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply demote_calls_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply demote_calls_in_handler_scoped | exact IH ].
    + apply demote_call_scoped.
  - destruct h as [params metadata handler].
    cbn [demote_calls_in_handler].
    apply scoped_handler_intro, demote_calls_scoped.
Qed.

Fixpoint demote_in_stubs_scoped (ids : list code_id) (e : expr) {struct e} :
  scoped e (demote_in_stubs ids e)
with demote_in_stubs_of_handler_scoped (ids : list code_id)
    (h : continuation_handler) {struct h} :
  scoped_handler h (demote_in_stubs_of_handler ids h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [demote_in_stubs]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply demote_in_stubs_scoped ].
    + apply scoped_let_static; [ | apply demote_in_stubs_scoped ].
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map]; [ apply scoped_bindings_nil | | | ].
      * cbn [demote_in_stub_binding code_is_stub].
        destruct (cm_is_stub metadata);
          [ | apply scoped_bindings_cons_same, IH ].
        cbn [demote_calls_in_code].
        apply scoped_bindings_cons_code; [ | exact IH ].
        apply scoped_code_intro, demote_calls_scoped.
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_same, IH.
    + apply scoped_let_cont;
        [ apply demote_in_stubs_of_handler_scoped
        | apply demote_in_stubs_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply demote_in_stubs_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply demote_in_stubs_of_handler_scoped | exact IH ].
  - destruct h as [params metadata handler].
    cbn [demote_in_stubs_of_handler].
    apply scoped_handler_intro, demote_in_stubs_scoped.
Qed.

Theorem demote_stub_calls_keeps_scope :
  demote_stub_calls_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e. unfold demote_stub_calls.
  apply demote_in_stubs_scoped.
Qed.

(* ======================================================================== *)
(* The removal of value slots                                               *)
(* ======================================================================== *)

Lemma drop_slots_static_const_free (dropped : list value_slot)
    (const : static_const) :
  incl (free_variables_of_static_const (drop_slots_static_const dropped const))
    (free_variables_of_static_const const).
Proof.
  destruct const;
    cbn [drop_slots_static_const free_variables_of_static_const];
    try apply incl_refl;
    unfold free_variables_of_value_slots, keep_slots;
    apply incl_flat_map_filter.
Qed.

Fixpoint drop_slots_scoped (dropped : list value_slot) (e : expr) {struct e} :
  scoped e (drop_slots dropped e)
with drop_slots_handler_scoped (dropped : list value_slot)
    (h : continuation_handler) {struct h} :
  scoped_handler h (drop_slots_handler dropped h)
with drop_slots_code_scoped (dropped : list value_slot) (c : code)
    {struct c} :
  scoped_code c (drop_slots_code dropped c).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [drop_slots]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply drop_slots_scoped ].
    + apply scoped_let_static; [ | apply drop_slots_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ apply scoped_bindings_nil | ].
      destruct binding as [id c | id | sym const]; cbn [drop_slots_binding].
      * apply scoped_bindings_cons_code;
          [ apply drop_slots_code_scoped | exact IH ].
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_symbol;
          [ apply drop_slots_static_const_free | exact IH ].
    + apply scoped_let_cont;
        [ apply drop_slots_handler_scoped | apply drop_slots_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply drop_slots_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply drop_slots_handler_scoped | exact IH ].
  - destruct h as [params metadata handler]. cbn [drop_slots_handler].
    apply scoped_handler_intro, drop_slots_scoped.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [drop_slots_code]. apply scoped_code_intro, drop_slots_scoped.
Qed.

Theorem drop_slots_keeps_scope : drop_slots_keeps_scope_statement.
Proof.
  intros dropped. apply keeps_scope_intro. intros e. apply drop_slots_scoped.
Qed.

Theorem without_slots_keeps_scope : without_slots_keeps_scope_statement.
Proof.
  intros slots. apply keeps_scope_intro. intros e.
  unfold without_slots, drop_unused_slots.
  destruct (slots_unused slots e); [ | apply scoped_refl ].
  apply (scoped_trans _ _ _ (drop_slots_scoped slots e)).
  apply (scoped_trans _ _ _ (remove_unused_lets_scoped _)).
  exact (rewrite_code_scoped remove_unused_lets
           remove_unused_lets_keeps_scope _).
Qed.

(* ======================================================================== *)
(* What is known of the static closures in sight                            *)
(* ======================================================================== *)

(* The variables of the constants that are known. *)
Definition known_closures_variables (known : known_closures)
    : list variable :=
  flat_map (fun entry => free_variables_of_static_const (snd entry)) known.

Lemma lookup_In {A : Type} (name : positive) (a : A)
    (bindings : list (positive * A)) :
  lookup name bindings = Some a -> In (name, a) bindings.
Proof.
  induction bindings as [ | [bound b] bindings IH]; cbn [lookup];
    [ discriminate | ].
  destruct (Pos.eqb name bound) eqn:Htest.
  - intros [= ->]. apply Pos.eqb_eq in Htest. subst bound. left.
    reflexivity.
  - intros Hlookup. right. exact (IH Hlookup).
Qed.

(* What a slot of a known closure holds is a variable of what is
   known, if it is a variable. *)
Lemma slot_read_free (known : known_closures)
    (project_from : option function_slot) (slot : value_slot)
    (closure contents : simple) :
  slot_read known project_from slot closure = Some contents ->
  incl (free_variables_of_simple contents) (known_closures_variables known).
Proof.
  unfold slot_read. destruct closure as [x | s | c]; try discriminate.
  destruct (lookup s known) as [const | ] eqn:Hknown; [ | discriminate ].
  destruct (static_closure_slots const) as [[at_slot slots] | ] eqn:Hslots;
    [ | discriminate ].
  intros Hread.
  assert (Hcontents : lookup slot slots = Some contents).
  { destruct project_from as [from | ]; [ | exact Hread ].
    destruct (Pos.eqb at_slot from); [ exact Hread | discriminate ]. }
  intros y Hy. unfold known_closures_variables. apply in_flat_map.
  exists (s, const). split; [ apply lookup_In, Hknown | ].
  cbn [snd].
  assert (Hin : In y (free_variables_of_value_slots slots)).
  { unfold free_variables_of_value_slots. apply in_flat_map.
    exists (slot, contents).
    split; [ apply lookup_In, Hcontents | exact Hy ]. }
  destruct const; cbn [static_closure_slots] in Hslots; try discriminate;
    injection Hslots as _ <-; exact Hin.
Qed.

Lemma read_slot_named_free (known : known_closures) (names : list variable)
    (d : named) :
  incl (known_closures_variables known) names ->
  incl (free_variables_of_named (read_slot_named known d))
    (free_variables_of_named d ++ names).
Proof.
  intros Hknown.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    cbn [read_slot_named]; try (apply incl_appl, incl_refl).
  - destruct (slot_read known None slot closure) as [contents | ] eqn:Hread;
      [ | apply incl_appl, incl_refl ].
    cbn [free_variables_of_named]. apply incl_appr.
    eapply incl_tran; [ exact (slot_read_free _ _ _ _ _ Hread) | ].
    exact Hknown.
  - destruct (slot_read known (Some project_from) slot closure)
      as [contents | ] eqn:Hread;
      [ | apply incl_appl, incl_refl ].
    cbn [free_variables_of_named]. apply incl_appr.
    eapply incl_tran; [ exact (slot_read_free _ _ _ _ _ Hread) | ].
    exact Hknown.
Qed.

Lemma forget_slot_var_variables (x : variable) (known : known_closures) :
  incl (known_closures_variables (forget_slot_var x known))
    (known_closures_variables known).
Proof.
  unfold known_closures_variables, forget_slot_var.
  apply incl_flat_map_filter.
Qed.

Lemma forget_slot_vars_variables (xs : list variable) :
  ∀ known : known_closures,
    incl (known_closures_variables (forget_slot_vars xs known))
      (known_closures_variables known).
Proof.
  induction xs as [ | x xs IH]; intros known; cbn [forget_slot_vars];
    [ apply incl_refl | ].
  eapply incl_tran; [ apply IH | apply forget_slot_var_variables ].
Qed.

Lemma static_closures_made_variables (bindings : list static_binding) :
  incl (known_closures_variables (flat_map static_closure_made bindings))
    (flat_map free_variables_of_binding bindings).
Proof.
  unfold known_closures_variables.
  induction bindings as [ | binding bindings IH]; cbn [flat_map];
    [ apply incl_refl | ].
  rewrite flat_map_app. apply incl_app_app; [ | exact IH ].
  destruct binding as [id c | id | s const]; cbn [static_closure_made];
    try (intros y []).
  destruct const; cbn [flat_map snd free_variables_of_binding];
    try (intros y []); rewrite app_nil_r; apply incl_refl.
Qed.

Fixpoint use_closures_knowing_scoped (e : expr) :
  ∀ (known : known_closures) (names : list variable),
    incl (known_closures_variables known) names ->
    scoped_under names e (use_closures_knowing known e).
Proof.
  intros known names Hknown.
  destruct e as [x kind d body | bindings body | k h body | handlers body
                | ap | ac | scrutinee arms | ];
    cbn [use_closures_knowing]; try apply scoped_under_refl.
  - apply scoped_under_let; [ apply read_slot_named_free, Hknown | ].
    apply use_closures_knowing_scoped.
    intros y Hy. right. apply in_or_app. left.
    exact (Hknown y (forget_slot_var_variables x known y Hy)).
  - apply scoped_under_let_static. apply use_closures_knowing_scoped.
    unfold closures_under_statics, known_closures_variables.
    rewrite flat_map_app.
    intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy];
      [ right; exact (static_closures_made_variables bindings y Hy)
      | left; exact (Hknown y Hy) ].
  - destruct h as [params metadata handler]. cbn [use_closures_in_handler].
    apply scoped_under_let_cont; apply use_closures_knowing_scoped;
      [ | exact Hknown ].
    eapply incl_tran; [ apply forget_slot_vars_variables | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply use_closures_knowing_scoped, Hknown.
  - assert (Hsame :
      free_variables_of_apply (call_known_closure known ap)
      = free_variables_of_apply ap
      ∧ ap_continuation (call_known_closure known ap) = ap_continuation ap
      ∧ ap_exn_continuation (call_known_closure known ap)
        = ap_exn_continuation ap).
    { unfold call_known_closure.
      destruct (ap_call_kind ap), (callee_code known (ap_callee ap));
        repeat split; reflexivity. }
    destruct Hsame as (Hv & Hret & Hexn).
    intros _. scope. rewrite Hv, Hret, Hexn.
    split; [ reflexivity | ].
    split; [ apply incl_appl, incl_refl | apply incl_refl ].
Qed.

Theorem use_known_closures_keeps_scope :
  use_known_closures_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e. apply scoped_under_none.
  apply use_closures_knowing_scoped. intros y [].
Qed.

(* ======================================================================== *)
(* The bindings of code moved up                                            *)
(* ======================================================================== *)

Lemma let_static_if_any_closed (bindings : list static_binding)
    (body : expr) :
  code_closed (let_static_if_any bindings body)
  = forallb binding_code_closed bindings && code_closed body.
Proof. destruct bindings; reflexivity. Qed.

Lemma let_static_if_any_free_variables (bindings : list static_binding)
    (body : expr) :
  free_variables_of_expr (let_static_if_any bindings body)
  = flat_map free_variables_of_binding bindings
    ++ free_variables_of_expr body.
Proof. destruct bindings; reflexivity. Qed.

Lemma let_static_if_any_free_continuations (bindings : list static_binding)
    (body : expr) :
  free_continuations_of_expr (let_static_if_any bindings body)
  = flat_map free_continuations_of_binding bindings
    ++ free_continuations_of_expr body.
Proof. destruct bindings; reflexivity. Qed.

(* The bindings of code of a group whose code is closed have no free
   variable. *)
Lemma code_bindings_closed_no_variables (bindings : list static_binding) :
  forallb binding_code_closed bindings = true ->
  flat_map free_variables_of_binding (filter binds_code bindings) = [].
Proof.
  induction bindings as [ | binding bindings IH]; cbn [forallb filter];
    [ reflexivity | ].
  intros Hclosed. apply andb_true_iff in Hclosed.
  destruct Hclosed as [Hfirst Hrest].
  destruct binding as [id c | id | s const];
    cbn [binds_code flat_map free_variables_of_binding app];
    [ | exact (IH Hrest) | exact (IH Hrest) ].
  rewrite (IH Hrest), app_nil_r.
  exact (proj1 (code_is_closed_free c Hfirst)).
Qed.

(* A binder that the bindings of code at the head of a term are moved
   over: the binder asks the same of its own parts whatever the term
   under it, and the free names of the whole grow with those of the
   term under it. *)
Lemma float_over_scoped (wrap : expr -> expr) (closed : bool)
    (inner : expr) :
  (∀ e : expr, code_closed (wrap e) = closed && code_closed e) ->
  (∀ e e' : expr,
    incl (free_variables_of_expr e') (free_variables_of_expr e) ->
    incl (free_variables_of_expr (wrap e'))
      (free_variables_of_expr (wrap e))) ->
  (∀ e e' : expr,
    incl (free_continuations_of_expr e') (free_continuations_of_expr e) ->
    incl (free_continuations_of_expr (wrap e'))
      (free_continuations_of_expr (wrap e))) ->
  scoped (wrap inner) (float_over wrap inner).
Proof.
  intros Hclosed Hv Hk.
  destruct inner as [x kind d body | bindings body | k h body
                    | handlers body | ap | ac | scrutinee arms | ];
    cbn [float_over]; try apply scoped_refl.
  intros Hwrap. rewrite Hclosed in Hwrap. scope_in Hwrap.
  apply andb_true_iff in Hwrap. destruct Hwrap as [Hwrap Hinner].
  apply andb_true_iff in Hinner. destruct Hinner as [Hbindings Hbody].
  rewrite let_static_if_any_closed, let_static_if_any_free_variables,
    let_static_if_any_free_continuations, Hclosed, let_static_if_any_closed,
    (code_bindings_closed_no_variables bindings Hbindings),
    (bindings_closed_no_continuations (filter binds_code bindings)
       (forallb_filter _ _ _ Hbindings)).
  rewrite !forallb_filter by exact Hbindings. rewrite Hwrap, Hbody.
  split; [ reflexivity | ]. cbn [app].
  split; [ apply Hv | apply Hk ];
    rewrite ?let_static_if_any_free_variables,
      ?let_static_if_any_free_continuations;
    scope;
    (apply incl_app_app; [ apply incl_flat_map_filter | apply incl_refl ]).
Qed.

Lemma float_over_let_scoped (x : variable) (kind : flambda_kind) (d : named)
    (inner : expr) :
  scoped (Let x kind d inner) (float_over (Let x kind d) inner).
Proof.
  apply (float_over_scoped (Let x kind d) true).
  - intros e. reflexivity.
  - intros e e' He. scope.
    apply incl_app_app; [ apply incl_refl | apply incl_without, He ].
  - intros e e' He. exact He.
Qed.

Lemma float_over_let_cont_scoped (k : continuation)
    (h : continuation_handler) (inner : expr) :
  scoped (Let_cont_non_recursive k h inner)
    (float_over (Let_cont_non_recursive k h) inner).
Proof.
  apply
    (float_over_scoped (Let_cont_non_recursive k h) (handler_code_closed h)).
  - intros e. reflexivity.
  - intros e e' He. scope. apply incl_app_app; [ apply incl_refl | exact He ].
  - intros e e' He. scope.
    apply incl_app_app; [ apply incl_refl | apply incl_without, He ].
Qed.

Lemma float_over_let_cont_recursive_scoped
    (handlers : list (continuation * continuation_handler)) (inner : expr) :
  scoped (Let_cont_recursive handlers inner)
    (float_over (Let_cont_recursive handlers) inner).
Proof.
  apply
    (float_over_scoped (Let_cont_recursive handlers)
       (forallb (fun '(_, h) => handler_code_closed h) handlers)).
  - intros e. reflexivity.
  - intros e e' He. scope. apply incl_app_app; [ apply incl_refl | exact He ].
  - intros e e' He. scope.
    apply incl_without, incl_app_app; [ apply incl_refl | exact He ].
Qed.

Fixpoint float_code_scoped (e : expr) : scoped e (float_code e).
Proof.
  destruct e as [x kind d body | bindings body
                | k [params metadata handler] body | handlers body | ap | ac
                | scrutinee arms | ];
    cbn [float_code]; try apply scoped_refl.
  - eapply scoped_trans;
      [ apply scoped_let; [ apply incl_refl | apply float_code_scoped ]
      | apply float_over_let_scoped ].
  - eapply scoped_trans;
      [ apply scoped_let_static;
        [ apply scoped_bindings_refl | apply (float_code_scoped body) ]
      | ].
    destruct (float_code body)
      as [x' kind' d' body' | bindings' body' | k' h' body'
         | handlers' body' | ap' | ac' | scrutinee' arms' | ];
      try
        (apply scoped_let_static;
         [ apply scoped_bindings_code_first | apply scoped_refl ]).
    eapply scoped_trans; [ apply scoped_let_static_merge | ].
    apply scoped_let_static;
      [ apply scoped_bindings_code_first | apply scoped_refl ].
  - eapply scoped_trans;
      [ apply scoped_let_cont;
        [ apply scoped_handler_intro, float_code_scoped
        | apply float_code_scoped ]
      | apply float_over_let_cont_scoped ].
  - eapply scoped_trans;
      [ apply scoped_let_cont_recursive;
        [ reflexivity | apply scoped_handlers_refl
        | apply float_code_scoped ]
      | apply float_over_let_cont_recursive_scoped ].
Qed.

Theorem float_code_keeps_scope : float_code_keeps_scope_statement.
Proof. apply keeps_scope_intro. exact float_code_scoped. Qed.

(* ======================================================================== *)
(* The groups moved above continuation bindings                             *)
(* ======================================================================== *)

(* A [Let_cont] binds no variable in its body, and a group whose code
   is closed mentions no continuation. *)
Lemma group_over_let_cont_scoped (k : continuation)
    (h : continuation_handler) (inner : expr) :
  scoped (Let_cont_non_recursive k h inner)
    (group_over (Let_cont_non_recursive k h) inner).
Proof.
  destruct inner as [x kind d body | bindings body | k' h' body
                    | handlers body | ap | ac | scrutinee arms | ];
    cbn [group_over]; try apply scoped_refl.
  intros Hclosed. scope_in Hclosed.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hh Hinner].
  apply andb_true_iff in Hinner. destruct Hinner as [Hbindings Hbody].
  scope.
  rewrite (bindings_closed_no_continuations bindings Hbindings), Hbindings,
    Hh, Hbody.
  split; [ reflexivity | ]. cbn [app].
  split; [ | apply incl_refl ].
  intros x Hx. rewrite !in_app_iff in *. tauto.
Qed.

Lemma group_over_let_cont_recursive_scoped
    (handlers : list (continuation * continuation_handler)) (inner : expr) :
  scoped (Let_cont_recursive handlers inner)
    (group_over (Let_cont_recursive handlers) inner).
Proof.
  destruct inner as [x kind d body | bindings body | k' h' body
                    | handlers' body | ap | ac | scrutinee arms | ];
    cbn [group_over]; try apply scoped_refl.
  intros Hclosed. scope_in Hclosed.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hh Hinner].
  apply andb_true_iff in Hinner. destruct Hinner as [Hbindings Hbody].
  scope.
  rewrite (bindings_closed_no_continuations bindings Hbindings), Hbindings,
    Hh, Hbody.
  split; [ reflexivity | ]. cbn [app].
  split; [ | apply incl_refl ].
  intros x Hx. rewrite !in_app_iff in *. tauto.
Qed.

Fixpoint groups_up_scoped (e : expr) : scoped e (groups_up e).
Proof.
  destruct e as [x kind d body | bindings body
                | k [params metadata handler] body | handlers body | ap | ac
                | scrutinee arms | ];
    cbn [groups_up]; try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | apply groups_up_scoped ].
  - eapply scoped_trans;
      [ apply scoped_let_static;
        [ apply scoped_bindings_refl | apply (groups_up_scoped body) ]
      | ].
    destruct (groups_up body)
      as [x' kind' d' body' | bindings' body' | k' h' body'
         | handlers' body' | ap' | ac' | scrutinee' arms' | ];
      try
        (apply scoped_let_static;
         [ apply scoped_bindings_code_first | apply scoped_refl ]).
    eapply scoped_trans; [ apply scoped_let_static_merge | ].
    apply scoped_let_static;
      [ apply scoped_bindings_code_first | apply scoped_refl ].
  - eapply scoped_trans;
      [ apply scoped_let_cont;
        [ apply scoped_handler_intro, groups_up_scoped
        | apply groups_up_scoped ]
      | apply group_over_let_cont_scoped ].
  - eapply scoped_trans;
      [ apply scoped_let_cont_recursive;
        [ reflexivity | apply scoped_handlers_refl
        | apply groups_up_scoped ]
      | apply group_over_let_cont_recursive_scoped ].
Qed.

Theorem groups_up_keeps_scope : groups_up_keeps_scope_statement.
Proof. apply keeps_scope_intro. exact groups_up_scoped. Qed.

(* ======================================================================== *)
(* The static bindings that need nothing bound first                        *)
(* ======================================================================== *)

(* A constant that has a value in the empty environment has no
   variable. *)
Lemma eval_simple_closed_free (s : simple) (v : value) :
  eval_simple [] s = Some v -> free_variables_of_simple s = [].
Proof.
  destruct s as [x | sym | c]; simpl; [ discriminate | | ]; reflexivity.
Qed.

Lemma eval_simples_closed_free (args : list simple) :
  ∀ vs : list value,
    eval_simples [] args = Some vs -> free_variables_of_simples args = [].
Proof.
  unfold free_variables_of_simples.
  induction args as [ | s args IH]; intros vs; simpl; [ reflexivity | ].
  destruct (eval_simple [] s) as [v | ] eqn:Hs; [ | discriminate ].
  destruct (eval_simples [] args) as [rest | ]; [ | discriminate ].
  intros _. rewrite (eval_simple_closed_free s v Hs), (IH rest eq_refl).
  reflexivity.
Qed.

Lemma eval_value_slots_closed_free (slots : list (value_slot * simple)) :
  ∀ vs : list (value_slot * value),
    eval_value_slots [] slots = Some vs ->
    free_variables_of_value_slots slots = [].
Proof.
  unfold free_variables_of_value_slots.
  induction slots as [ | [slot s] slots IH]; intros vs; simpl;
    [ reflexivity | ].
  destruct (eval_simple [] s) as [v | ] eqn:Hs; [ | discriminate ].
  destruct (eval_value_slots [] slots) as [rest | ]; [ | discriminate ].
  intros _. rewrite (eval_simple_closed_free s v Hs), (IH rest eq_refl).
  reflexivity.
Qed.

Lemma eval_static_const_closed_free (const : static_const)
    (allocated : statically_allocated_value) :
  eval_static_const [] const = Some allocated ->
  free_variables_of_static_const const = [].
Proof.
  destruct const as [code slots | slot syms decls slots | tag fields
                    | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots [] slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    intros _. exact (eval_value_slots_closed_free slots vs Hslots).
  - destruct (lookup slot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots [] slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    intros _. exact (eval_value_slots_closed_free slots vs Hslots).
  - destruct (eval_simples [] fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    intros _. exact (eval_simples_closed_free fields vs Hfields).
  - destruct (eval_simple [] contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    intros _. exact (eval_simple_closed_free contents v Hcontents).
  - intros _. reflexivity.
  - destruct (eval_simples [] elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    intros _. exact (eval_simples_closed_free elements vs Helements).
Qed.

(* The bindings that are bound first have no free name, where the code
   of the group is closed. *)
Lemma hoisted_of_group_no_free_names (bindings : list static_binding) :
  forallb binding_code_closed bindings = true ->
  no_free_names (filter is_hoisted bindings).
Proof.
  intros Hclosed.
  assert (Hfiltered :
    forallb binding_code_closed (filter is_hoisted bindings) = true)
    by (apply forallb_filter, Hclosed).
  split; [ exact Hfiltered | ].
  split; [ | exact (bindings_closed_no_continuations _ Hfiltered) ].
  clear Hfiltered.
  induction bindings as [ | binding bindings IH]; cbn [filter];
    [ reflexivity | ].
  cbn [forallb] in Hclosed. apply andb_true_iff in Hclosed.
  destruct Hclosed as [Hfirst Hrest].
  destruct (is_hoisted binding) eqn:Hhoisted; [ | exact (IH Hrest) ].
  cbn [flat_map]. rewrite (IH Hrest), app_nil_r.
  destruct binding as [id c | id | sym const]; scope_in Hfirst; scope.
  - exact (proj1 (code_is_closed_free c Hfirst)).
  - discriminate Hhoisted.
  - cbn [is_hoisted] in Hhoisted.
    destruct (eval_static_const [] const) as [allocated | ] eqn:Hconst;
      [ | discriminate Hhoisted ].
    exact (eval_static_const_closed_free const allocated Hconst).
Qed.

Fixpoint hoisted_bindings_no_free_names (e : expr) {struct e} :
  code_closed e = true -> no_free_names (hoisted_bindings e)
with handler_hoisted_bindings_no_free_names (h : continuation_handler)
    {struct h} :
  handler_code_closed h = true ->
  no_free_names (handler_hoisted_bindings h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [hoisted_bindings]; intros Hclosed; scope_in Hclosed;
      try solve [ repeat split ].
    + exact (hoisted_bindings_no_free_names body Hclosed).
    + apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
      apply no_free_names_app;
        [ exact (hoisted_of_group_no_free_names bindings Hfirst)
        | exact (hoisted_bindings_no_free_names body Hrest) ].
    + apply andb_true_iff in Hclosed. destruct Hclosed as [Hh Hbody].
      apply no_free_names_app;
        [ exact (hoisted_bindings_no_free_names body Hbody)
        | exact (handler_hoisted_bindings_no_free_names h Hh) ].
    + apply andb_true_iff in Hclosed. destruct Hclosed as [_ Hbody].
      exact (hoisted_bindings_no_free_names body Hbody).
  - destruct h as [params metadata handler].
    cbn [handler_hoisted_bindings]. intros Hclosed. scope_in Hclosed.
    exact (hoisted_bindings_no_free_names handler Hclosed).
Qed.

(* The unit with those bindings removed. *)
Fixpoint without_hoisted_scoped (e : expr) {struct e} :
  scoped e (without_hoisted e)
with handler_without_hoisted_scoped (h : continuation_handler) {struct h} :
  scoped_handler h (handler_without_hoisted h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [without_hoisted]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply without_hoisted_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_filter | apply without_hoisted_scoped ].
    + apply scoped_let_cont;
        [ apply handler_without_hoisted_scoped | apply without_hoisted_scoped ].
    + apply scoped_let_cont_recursive;
        [ reflexivity | apply scoped_handlers_refl
        | apply without_hoisted_scoped ].
  - destruct h as [params metadata handler].
    cbn [handler_without_hoisted].
    apply scoped_handler_intro, without_hoisted_scoped.
Qed.

Theorem with_statics_first_keeps_scope :
  with_statics_first_keeps_scope_statement.
Proof.
  intros Ls. apply keeps_scope_intro. intros e.
  unfold with_statics_first.
  destruct Ls as [ | L Ls]; [ apply scoped_refl | ].
  unfold hoist_statics.
  match goal with
  | |- scoped _ (match (if ?c then _ else _) with _ => _ end) => destruct c
  end;
    [ | apply scoped_refl ].
  eapply scoped_trans;
    [ | eapply scoped_trans;
        [ apply drop_deleted_code_scoped | apply merge_statics_scoped ] ].
  intros Hclosed.
  exact (scoped_let_static_around _ _ _
           (hoisted_bindings_no_free_names e Hclosed)
           (without_hoisted_scoped e) Hclosed).
Qed.

(* ======================================================================== *)
(* A parameter that every application gives one variable                    *)
(* ======================================================================== *)

Theorem unalias_params_keeps_scope : unalias_params_keeps_scope_statement.
Proof. exact FlambdaUnaliasParamsNamesProofs.unalias_params_scope. Qed.

(* ======================================================================== *)
(* A parameter that every application gives the number of its arm           *)
(* ======================================================================== *)

Theorem give_scrutinees_keeps_scope : give_scrutinees_keeps_scope_statement.
Proof. exact FlambdaGiveScrutineeNamesProofs.give_scrutinees_scope. Qed.

(* ======================================================================== *)
(* The scrutinee is the number of the arm                                   *)
(* ======================================================================== *)

Theorem numbers_in_arms_keeps_scope : numbers_in_arms_keeps_scope_statement.
Proof. exact FlambdaArmNumbersNamesProofs.numbers_in_arms_scope. Qed.

(* ======================================================================== *)
(* An unused parameter of a continuation that a call returns to             *)
(* ======================================================================== *)

Theorem wrap_returns_keeps_scope : wrap_returns_keeps_scope_statement.
Proof. exact FlambdaWrapReturnsNamesProofs.wrap_returns_scope. Qed.

(* ======================================================================== *)
(* Parameters that are only untagged or passed on                           *)
(* ======================================================================== *)

Theorem untag_params_keeps_scope : untag_params_keeps_scope_statement.
Proof. exact FlambdaUntagParamsNamesProofs.untag_params_scope. Qed.

(* ======================================================================== *)
(* Parameters bound again at the head of their handlers, and their twins    *)
(* ======================================================================== *)

Theorem shadow_params_keeps_scope : shadow_params_keeps_scope_statement.
Proof. exact FlambdaShadowParamsNamesProofs.shadow_params_scope. Qed.

Theorem twin_params_keeps_scope : twin_params_keeps_scope_statement.
Proof. exact FlambdaTwinParamsNamesProofs.twin_params_scope. Qed.

(* ======================================================================== *)
(* A parameter that every application gives one symbol or one constant      *)
(* ======================================================================== *)

Theorem bind_const_params_keeps_scope :
  bind_const_params_keeps_scope_statement.
Proof. exact FlambdaBindConstParamsNamesProofs.bind_const_params_scope. Qed.

(* ======================================================================== *)
(* A parameter of a loop that every turn passes on                          *)
(* ======================================================================== *)

Theorem unalias_rec_params_keeps_scope :
  unalias_rec_params_keeps_scope_statement.
Proof.
  exact FlambdaUnaliasRecParamsNamesProofs.unalias_rec_params_scope.
Qed.
