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

(* Proofs of the statements of FlambdaUntagAtBinderSpec.v, which has
   the definitions, the statements and what they depend on: the
   test of decision D25, at parameters. The proofs are of the machine
   that tests what a binder states of its parameters where a handler
   is entered.

   This file needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaContextMemoryProofs
  FlambdaClosuresProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaHeapSpec FlambdaCseSpec FlambdaCseProofs
  FlambdaSubkindSpec FlambdaSubkindOrderSpec FlambdaUntagParamsSpec
  FlambdaUntagAtBinderSpec.
Import ListNotations.

(* ======================================================================== *)
(* The value that a binder accepts                                          *)
(* ======================================================================== *)

Lemma stated_is_tagged (statics : static_env) (ps : param_subkind)
    (v : value) :
  states_tagged ps = true ->
  value_at statics ps v = true ->
  is_tagged_immediate v = true.
Proof.
  destruct ps as [nullable sk].
  unfold states_tagged, param_says_at_least, value_at. cbn [fst snd].
  intros Hstates Hat.
  apply andb_true_iff in Hstates as [Hnull Hsk].
  destruct nullable; cbn in Hnull; [ discriminate Hnull | ].
  destruct sk as [ | | n | consts shapes]; cbn in Hsk;
    try discriminate Hsk.
  - destruct v as [c | code slots | slot decls slots | s | tag fields
      | elements | contents | l | callee args args_arity];
      [ destruct c | .. ]; cbn in Hat |- *;
      first [ discriminate Hat | exact Hat ].
  - destruct shapes; cbn in Hsk; try discriminate Hsk.
    destruct v as [c | code slots | slot decls slots | s | tag fields
      | elements | contents | l | callee args args_arity];
      [ destruct c | .. ]; cbn in Hat;
      try discriminate Hat;
      try (apply andb_true_iff in Hat as [Hat _]; exact Hat);
      try (repeat
             match type of Hat with
             | context [match ?x with _ => _ end] => destruct x
             end;
           discriminate Hat).
Qed.

Theorem untagging_of_stated : untagging_of_stated_statement.
Proof.
  intros statics ps v Hstates Hat.
  pose proof (stated_is_tagged statics ps v Hstates Hat) as Htagged.
  unfold is_tagged_immediate in Htagged.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ] eqn:Hint;
    [ | discriminate Htagged ].
  exists i.
  assert
    (Hstep :
      eval_unary_primitive statics Untag_immediate v
      = match int_of_value Standard_tagged_immediate v with
        | Some i => Some (Value_const (Naked_immediate i))
        | None => None
        end) by reflexivity.
  rewrite Hstep, Hint. reflexivity.
Qed.

(* ======================================================================== *)
(* What an untagging is, and what it gives                                  *)
(* ======================================================================== *)

Lemma untags_spec (x : variable) (kind : flambda_kind) (d : named) :
  untags x kind d = true ->
  kind = untagged_kind ∧ d = Prim (Unary Untag_immediate) [Var x].
Proof.
  unfold untags. intros H. apply andb_true_iff in H as [Hkind Hd]. split.
  - unfold kind_eqb in Hkind.
    destruct (kind_eq_dec kind untagged_kind) as [Heq | _];
      [ exact Heq | discriminate Hkind ].
  - destruct (untag_of d) as [p | ] eqn:Huntag; [ | discriminate Hd ].
    apply Pos.eqb_eq in Hd. subst p.
    unfold untag_of in Huntag.
    destruct d as [ | p args | | | | | ]; cbn in Huntag;
      try discriminate Huntag.
    destruct p as [q | | | | ]; cbn in Huntag; try discriminate Huntag.
    destruct q; cbn in Huntag; try discriminate Huntag.
    destruct args as [ | [y | | ] [ | ]]; cbn in Huntag;
      try discriminate Huntag.
    injection Huntag as ->. reflexivity.
Qed.

(* The untagging of a variable that holds a value on which it has one
   runs, as a [Let] runs it, to that number, and leaves the heap and
   the record as they are. *)
Lemma untag_exec (statics : static_env) (hp : heap) (rd : read_record)
    (vars : env) (x : variable) (v : value) (i : Z) :
  lookup x vars = Some v ->
  eval_unary_primitive statics Untag_immediate v
  = Some (Value_const (Naked_immediate i)) ->
  exec_named statics hp rd vars (Prim (Unary Untag_immediate) [Var x])
  = Some (Value_const (Naked_immediate i), hp, rd).
Proof.
  intros Hlookup Huntag.
  apply
    (eval_named_exec_named statics hp rd vars (Unary Untag_immediate)
       [Var x] [v] (Value_const (Naked_immediate i))).
  - reflexivity.
  - cbn [eval_simples eval_simple]. rewrite Hlookup. reflexivity.
  - intros l [Hl | []]. subst v. cbn in Huntag. discriminate Huntag.
  - rewrite eval_named_prim. cbn [eval_simples eval_simple].
    rewrite Hlookup. exact Huntag.
Qed.

(* What an untagging that has run gave is a naked immediate. *)
Lemma untag_exec_value (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (x : variable) (vu : value) :
  exec_named statics hp rd vars (Prim (Unary Untag_immediate) [Var x])
  = Some (vu, hp', rd') ->
  ∃ i : Z, vu = Value_const (Naked_immediate i).
Proof.
  intros Hexec.
  destruct
    (exec_named_eval_named statics hp hp' rd rd' vars
       (Unary Untag_immediate) [Var x] vu eq_refl Hexec)
    as (Heval & _ & _).
  rewrite eval_named_prim in Heval.
  cbn [eval_simples eval_simple] in Heval.
  destruct (lookup x vars) as [v | ]; [ | discriminate Heval ].
  assert
    (Huntag :
      match int_of_value Standard_tagged_immediate v with
      | Some i => Some (Value_const (Naked_immediate i))
      | None => None
      end
      = Some vu) by exact Heval.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ];
    [ | discriminate Huntag ].
  exists i. injection Huntag as <-. reflexivity.
Qed.

(* ======================================================================== *)
(* What is known where an untagging has run                                 *)
(* ======================================================================== *)

(* [u] holds the untagging of [x]: what FlambdaCseSpec.all_computed
   says of one entry. *)
Definition untagged_known (x u : variable) : computed :=
  [(u, (Unary Untag_immediate, [Var x]))].

Lemma untagged_known_forget (x u y : variable) :
  Pos.eqb y u = false ->
  Pos.eqb y x = false ->
  forget_computed y (untagged_known x u) = untagged_known x u.
Proof.
  intros Hu Hx. unfold forget_computed, untagged_known. cbn.
  rewrite Hu, Hx. reflexivity.
Qed.

Lemma untagged_known_forget_vars (x u : variable) (params : list variable) :
  among x params = false ->
  among u params = false ->
  forget_computed_vars params (untagged_known x u) = untagged_known x u.
Proof.
  induction params as [ | p params IH]; intros Hx Hu; [ reflexivity | ].
  rewrite among_cons in Hx, Hu.
  apply orb_false_iff in Hx as [Hxp Hx]. apply orb_false_iff in Hu as [Hup Hu].
  cbn [forget_computed_vars].
  rewrite untagged_known_forget;
    [ exact (IH Hx Hu) | rewrite Pos.eqb_sym; exact Hup
    | rewrite Pos.eqb_sym; exact Hxp ].
Qed.

(* ======================================================================== *)
(* The [Let]s taken away                                                    *)
(* ======================================================================== *)

Fixpoint drop_untagging_binds_statics (e : expr) :
  ∀ x u : variable, binds_statics (drop_untagging x u e) = binds_statics e
with drop_untagging_in_handler_binds_statics (h : continuation_handler) :
  ∀ x u : variable,
    handler_binds_statics (drop_untagging_in_handler x u h)
    = handler_binds_statics h.
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros x u; simpl; try reflexivity.
    + destruct (Pos.eqb y u).
      * destruct (untags x kind d);
          [ apply drop_untagging_binds_statics | reflexivity ].
      * destruct (Pos.eqb y x); [ reflexivity | ].
        simpl. apply drop_untagging_binds_statics.
    + rewrite drop_untagging_binds_statics. reflexivity.
    + rewrite drop_untagging_in_handler_binds_statics,
        drop_untagging_binds_statics.
      reflexivity.
    + rewrite drop_untagging_binds_statics. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite drop_untagging_in_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler]. intros x u. simpl.
    destruct (among x params || among u params); [ reflexivity | ].
    simpl. apply drop_untagging_binds_statics.
Qed.

Lemma drop_untagging_in_handlers_bind_statics (x u : variable)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, drop_untagging_in_handler x u h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite drop_untagging_in_handler_binds_statics, IH. reflexivity.
Qed.

(* A [Let] of [u] to the untagging of [x], where [u] holds it already:
   its step binds [u] to what it held. *)
Lemma known_let_equivalent (x u : variable) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record) :
  all_computed (untagged_known x u) vars statics hp rd ->
  equivalent
    (State
       (Let u untagged_kind (Prim (Unary Untag_immediate) [Var x]) body)
       vars conts statics hp rd)
    (State body vars conts statics hp rd).
Proof.
  intros Hhold.
  destruct (Hhold _ (or_introl eq_refl)) as (_ & vu & Hexec & Hlookup).
  cbn [fst snd untagged_known] in Hexec, Hlookup.
  destruct (untag_exec_value statics hp hp rd rd vars x vu Hexec) as [i ->].
  apply
    (equivalent_trans _
       (State body ((u, Value_const (Naked_immediate i)) :: vars) conts
          statics hp rd)).
  - apply step_equivalent. rewrite step_let_exec, Hexec. reflexivity.
  - apply equivalent_sym, coincide_equivalent.
    + intros y _. simpl.
      destruct (Pos.eqb_spec y u) as [-> | _];
        [ symmetry; exact Hlookup | reflexivity ].
    + intros k _. apply (conts_agree_same identity_renaming). reflexivity.
Qed.

(* The term refines the term without the [Let]s. *)
Fixpoint drop_untagging_refines (e : expr) :
  ∀ (x u : variable) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    all_computed (untagged_known x u) vars statics hp rd ->
    refines (State e vars conts statics hp rd)
      (State (drop_untagging x u e) vars conts statics hp rd)
with drop_untagging_in_handler_refines (h : continuation_handler) :
  ∀ (x u : variable) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    all_computed (untagged_known x u) vars statics hp rd ->
    refines
      (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (drop_untagging_in_handler x u h) body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros x u vars conts statics hp rd Hhold; simpl.
    + (* Let *)
      destruct (Pos.eqb y u) eqn:Hyu.
      * destruct (untags x kind d) eqn:Huntags; [ | apply refines_refl ].
        apply Pos.eqb_eq in Hyu. subst y.
        destruct (untags_spec x kind d Huntags) as [-> ->].
        apply
          (refines_trans _ (State body vars conts statics hp rd)).
        -- apply equivalent_refines, known_let_equivalent, Hhold.
        -- apply drop_untagging_refines, Hhold.
      * destruct (Pos.eqb y x) eqn:Hyx; [ apply refines_refl | ].
        apply let_refines_step_exec; [ reflexivity | ].
        intros v hp' rd' Hd _.
        apply drop_untagging_refines.
        rewrite <- (untagged_known_forget x u y Hyu Hyx).
        apply all_computed_forget.
        exact
          (all_computed_evolves _ vars statics hp hp' rd rd' Hhold
             (exec_named_memory_evolves statics hp hp' rd rd' vars d v Hd)).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply drop_untagging_refines.
      exact
        (all_computed_extend _ vars statics extended hp rd Hhold
           (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (drop_untagging_in_handler x u h) body)
           vars conts statics hp rd)).
      * apply drop_untagging_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step, drop_untagging_refines,
          Hhold.
    + (* Let_cont_recursive: the handlers, all at once; then the body. *)
      apply (refines_trans _
        (State
           (Let_cont_recursive
              (map (fun '(k, h) => (k, drop_untagging_in_handler x u h))
                 handlers)
              body)
           vars conts statics hp rd)).
      * apply
          (let_cont_recursive_handlers_refine_where_memory
             (all_computed (untagged_known x u) vars)
             (all_computed_extend (untagged_known x u) vars)
             (all_computed_evolves (untagged_known x u) vars));
          [ exact Hhold
          | symmetry; apply drop_untagging_in_handlers_bind_statics
          | | ];
          (induction handlers as [ | [k [params metadata handler]] handlers IH];
             cbn [map drop_untagging_in_handler];
             [ apply Forall2_nil | ];
           apply Forall2_cons; [ | exact IH ];
           split; [ reflexivity | ]; cbn [snd];
           destruct (among x params || among u params) eqn:Hp;
           apply Handler_related;
           intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind;
           [ apply refines_refl | ];
           apply orb_false_iff in Hp as [Hx Hu];
           apply drop_untagging_refines;
           rewrite <- (untagged_known_forget_vars x u params Hx Hu);
           exact
             (all_computed_bind_params statics' hp' rd' params
                (untagged_known x u) vs vars handler_vars Hhold' Hbind)).
      * apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
        apply drop_untagging_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros x u k body vars conts statics hp rd Hhold. simpl.
    destruct (among x params || among u params) eqn:Hp;
      [ apply refines_refl | ].
    apply orb_false_iff in Hp as [Hx Hu].
    apply
      (let_cont_handler_refines_where_memory
         (all_computed (untagged_known x u) vars)
         (all_computed_extend (untagged_known x u) vars)
         (all_computed_evolves (untagged_known x u) vars));
      [ exact Hhold | ].
    intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind.
    apply drop_untagging_refines.
    rewrite <- (untagged_known_forget_vars x u params Hx Hu).
    exact
      (all_computed_bind_params statics' hp' rd' params (untagged_known x u)
         vs vars handler_vars Hhold' Hbind).
Qed.

(* The term without the [Let]s refines the term. *)
Fixpoint drop_untagging_refined (e : expr) :
  ∀ (x u : variable) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    all_computed (untagged_known x u) vars statics hp rd ->
    refines (State (drop_untagging x u e) vars conts statics hp rd)
      (State e vars conts statics hp rd)
with drop_untagging_in_handler_refined (h : continuation_handler) :
  ∀ (x u : variable) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    all_computed (untagged_known x u) vars statics hp rd ->
    refines
      (State
         (Let_cont_non_recursive k (drop_untagging_in_handler x u h) body)
         vars conts statics hp rd)
      (State (Let_cont_non_recursive k h body) vars conts statics hp rd).
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros x u vars conts statics hp rd Hhold; simpl.
    + (* Let *)
      destruct (Pos.eqb y u) eqn:Hyu.
      * destruct (untags x kind d) eqn:Huntags; [ | apply refines_refl ].
        apply Pos.eqb_eq in Hyu. subst y.
        destruct (untags_spec x kind d Huntags) as [-> ->].
        apply
          (refines_trans _ (State body vars conts statics hp rd)).
        -- apply drop_untagging_refined, Hhold.
        -- apply equivalent_refines, equivalent_sym, known_let_equivalent,
             Hhold.
      * destruct (Pos.eqb y x) eqn:Hyx; [ apply refines_refl | ].
        apply let_refines_step_exec; [ reflexivity | ].
        intros v hp' rd' Hd _.
        apply drop_untagging_refined.
        rewrite <- (untagged_known_forget x u y Hyu Hyx).
        apply all_computed_forget.
        exact
          (all_computed_evolves _ vars statics hp hp' rd rd' Hhold
             (exec_named_memory_evolves statics hp hp' rd rd' vars d v Hd)).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply drop_untagging_refined.
      exact
        (all_computed_extend _ vars statics extended hp rd Hhold
           (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (drop_untagging_in_handler x u h) body)
           vars conts statics hp rd)).
      * apply let_cont_non_recursive_refines_step, drop_untagging_refined,
          Hhold.
      * apply drop_untagging_in_handler_refined, Hhold.
    + (* Let_cont_recursive: the body; then the handlers, all at once. *)
      apply (refines_trans _
        (State
           (Let_cont_recursive
              (map (fun '(k, h) => (k, drop_untagging_in_handler x u h))
                 handlers)
              body)
           vars conts statics hp rd)).
      * apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
        apply drop_untagging_refined, Hhold.
      * apply
          (let_cont_recursive_handlers_refine_where_memory
             (all_computed (untagged_known x u) vars)
             (all_computed_extend (untagged_known x u) vars)
             (all_computed_evolves (untagged_known x u) vars));
          [ exact Hhold
          | apply drop_untagging_in_handlers_bind_statics
          | | ];
          (induction handlers as [ | [k [params metadata handler]] handlers IH];
             cbn [map drop_untagging_in_handler];
             [ apply Forall2_nil | ];
           apply Forall2_cons; [ | exact IH ];
           split; [ reflexivity | ]; cbn [snd];
           destruct (among x params || among u params) eqn:Hp;
           apply Handler_related;
           intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind;
           [ apply refines_refl | ];
           apply orb_false_iff in Hp as [Hx Hu];
           apply drop_untagging_refined;
           rewrite <- (untagged_known_forget_vars x u params Hx Hu);
           exact
             (all_computed_bind_params statics' hp' rd' params
                (untagged_known x u) vs vars handler_vars Hhold' Hbind)).
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros x u k body vars conts statics hp rd Hhold. simpl.
    destruct (among x params || among u params) eqn:Hp;
      [ apply refines_refl | ].
    apply orb_false_iff in Hp as [Hx Hu].
    apply
      (let_cont_handler_refines_where_memory
         (all_computed (untagged_known x u) vars)
         (all_computed_extend (untagged_known x u) vars)
         (all_computed_evolves (untagged_known x u) vars));
      [ exact Hhold | ].
    intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind.
    apply drop_untagging_refined.
    rewrite <- (untagged_known_forget_vars x u params Hx Hu).
    exact
      (all_computed_bind_params statics' hp' rd' params (untagged_known x u)
         vs vars handler_vars Hhold' Hbind).
Qed.

(* ======================================================================== *)
(* A parameter that states a tagged immediate holds one                     *)
(* ======================================================================== *)

Lemma tagged_params_among (x : variable) :
  ∀ (params : list variable) (sks : list param_subkind),
    x ∈ tagged_params params sks -> x ∈ params.
Proof.
  induction params as [ | p params IH]; intros [ | ps sks] Hin;
    simpl in Hin; try contradiction.
  destruct (states_tagged ps).
  - destruct Hin as [-> | Hin]; [ left; reflexivity | right ].
    exact (IH sks Hin).
  - right. exact (IH sks Hin).
Qed.

Lemma tagged_param_bound (statics : static_env) (x : variable) :
  ∀ (params : list variable) (sks : list param_subkind) (vs : list value)
    (vars handler_vars : env),
    distinct params = true ->
    x ∈ tagged_params params sks ->
    values_of_subkinds statics vs sks = true ->
    bind_params params vs vars = Some handler_vars ->
    ∃ (ps : param_subkind) (v : value),
      states_tagged ps = true
      ∧ value_at statics ps v = true
      ∧ lookup x handler_vars = Some v.
Proof.
  induction params as [ | p params IH];
    intros [ | ps sks] [ | v vs] vars handler_vars Hdistinct Hin Hvalues
      Hbind;
    simpl in Hin; try contradiction; simpl in Hbind; try discriminate Hbind.
  cbn [distinct] in Hdistinct.
  apply andb_true_iff in Hdistinct as [Hfirst Hdistinct].
  apply negb_true_iff in Hfirst.
  cbn [values_of_subkinds] in Hvalues.
  apply andb_true_iff in Hvalues as [Hat Hvalues].
  destruct (states_tagged ps) eqn:Hstates.
  - destruct Hin as [<- | Hin].
    + exists ps, v. split; [ exact Hstates | ]. split; [ exact Hat | ].
      rewrite
        (bind_params_lookup_other p params vs ((p, v) :: vars) handler_vars
           Hfirst Hbind).
      simpl. rewrite Pos.eqb_refl. reflexivity.
    + exact (IH sks vs ((p, v) :: vars) handler_vars Hdistinct Hin Hvalues
               Hbind).
  - exact (IH sks vs ((p, v) :: vars) handler_vars Hdistinct Hin Hvalues
             Hbind).
Qed.

(* ======================================================================== *)
(* The untagging at the head of a handler                                   *)
(* ======================================================================== *)

Lemma untagging_at_head_equivalent (params : list variable)
    (metadata : handler_metadata) (def_vars : env) (x : variable)
    (handler : expr) (vs : list value) (handler_vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record) :
  distinct params = true ->
  x ∈ tagged_params params (hm_params_subkinds metadata) ->
  values_of_subkinds statics vs (hm_params_subkinds metadata) = true ->
  bind_params params vs def_vars = Some handler_vars ->
  equivalent (State handler handler_vars conts statics hp rd)
    (State (untagging_at_head params x handler) handler_vars conts statics hp
       rd).
Proof.
  intros Hdistinct Hin Hvalues Hbind. unfold untagging_at_head.
  destruct (first_untagging x handler) as [u | ]; [ | apply equivalent_refl ].
  destruct (negb (among u params) && negb (var_free_b u handler)) eqn:Hok;
    [ | apply equivalent_refl ].
  apply andb_true_iff in Hok as [Hparam Hfree].
  apply negb_true_iff in Hparam. apply negb_true_iff in Hfree.
  destruct
    (tagged_param_bound statics x params (hm_params_subkinds metadata) vs
       def_vars handler_vars Hdistinct Hin Hvalues Hbind)
    as (ps & v & Hstates & Hat & Hlookup).
  destruct (untagging_of_stated statics ps v Hstates Hat) as [i Huntag].
  assert (Hxu : Pos.eqb x u = false).
  { destruct (Pos.eqb_spec x u) as [Heq | _]; [ | reflexivity ].
    subst u.
    apply (tagged_params_among x params) in Hin. apply among_In in Hin.
    congruence. }
  assert
    (Hknown :
      all_computed (untagged_known x u)
        ((u, Value_const (Naked_immediate i)) :: handler_vars) statics hp rd).
  { intros entry [<- | []].
    split; [ apply shareable_stable; reflexivity | ].
    exists (Value_const (Naked_immediate i)). cbn [fst snd]. split.
    - apply (untag_exec statics hp rd _ x v i); [ | exact Huntag ].
      simpl. rewrite Hxu. exact Hlookup.
    - simpl. rewrite Pos.eqb_refl. reflexivity. }
  apply
    (equivalent_trans _
       (State handler ((u, Value_const (Naked_immediate i)) :: handler_vars)
          conts statics hp rd)).
  - apply coincide_equivalent.
    + intros y Hy. simpl.
      destruct (Pos.eqb_spec y u) as [-> | _]; [ | reflexivity ].
      apply var_free_b_sound in Hy. congruence.
    + intros k _. apply (conts_agree_same identity_renaming). reflexivity.
  - apply
      (equivalent_trans _
         (State (drop_untagging x u handler)
            ((u, Value_const (Naked_immediate i)) :: handler_vars) conts
            statics hp rd)).
    + apply refines_antisymmetric;
        [ apply drop_untagging_refines | apply drop_untagging_refined ];
        exact Hknown.
    + apply equivalent_sym, step_equivalent.
      rewrite step_let_exec,
        (untag_exec statics hp rd handler_vars x v i Hlookup Huntag).
      reflexivity.
Qed.

(* The untaggings of several parameters, one after the other. *)
Lemma untaggings_at_head_equivalent (params : list variable)
    (metadata : handler_metadata) (def_vars : env) (vs : list value)
    (handler_vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  distinct params = true ->
  values_of_subkinds statics vs (hm_params_subkinds metadata) = true ->
  bind_params params vs def_vars = Some handler_vars ->
  ∀ (xs : list variable) (handler : expr),
    (∀ x : variable,
      x ∈ xs -> x ∈ tagged_params params (hm_params_subkinds metadata)) ->
    equivalent (State handler handler_vars conts statics hp rd)
      (State (fold_right (untagging_at_head params) handler xs) handler_vars
         conts statics hp rd).
Proof.
  intros Hdistinct Hvalues Hbind.
  induction xs as [ | x xs IH]; intros handler Hxs; simpl;
    [ apply equivalent_refl | ].
  apply
    (equivalent_trans _
       (State (fold_right (untagging_at_head params) handler xs) handler_vars
          conts statics hp rd)).
  - apply IH. intros y Hy. apply Hxs. right. exact Hy.
  - apply
      (untagging_at_head_equivalent params metadata def_vars x _ vs
         handler_vars conts statics hp rd Hdistinct);
      [ apply Hxs; left; reflexivity | exact Hvalues | exact Hbind ].
Qed.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

Lemma expr_equivalent_same (e : expr) : expr_equivalent e e.
Proof. intros vars conts statics hp rd. apply equivalent_refl. Qed.

Fixpoint bind_untaggings_equivalent_aux (e : expr) :
  expr_equivalent e (bind_untaggings e)
with bind_untaggings_in_handler_equivalent (h : continuation_handler) :
  ∀ (returned_to : bool) (k : continuation) (body : expr),
    expr_equivalent (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (bind_untaggings_in_handler returned_to h)
         body).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      simpl; try apply expr_equivalent_same.
    + exact
        (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
           (bind_untaggings_equivalent_aux body)).
    + exact
        (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
           (bind_untaggings_equivalent_aux body)).
    + apply
        (expr_equivalent_trans _ _ _
           (bind_untaggings_in_handler_equivalent h (returns_to k body) k
              body)).
      exact
        (plug_equivalent
           (Let_cont_non_recursive_body k
              (bind_untaggings_in_handler (returns_to k body) h) Hole)
           _ _ (not_many _) (bind_untaggings_equivalent_aux body)).
    + exact
        (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
           (not_many _) (bind_untaggings_equivalent_aux body)).
  - destruct h as [params metadata handler].
    intros returned_to k body. simpl.
    pose proof
      (plug_equivalent
         (Let_cont_non_recursive_handler k params metadata Hole body) _ _
         (not_many _) (bind_untaggings_equivalent_aux handler))
      as Hinner.
    simpl in Hinner.
    destruct
      (returned_to && negb (hm_is_exn_handler metadata) && distinct params)
      eqn:Hgo; [ | exact Hinner ].
    apply andb_true_iff in Hgo as [_ Hdistinct].
    apply (expr_equivalent_trans _ _ _ Hinner).
    intros vars conts statics hp rd.
    apply refines_antisymmetric; apply let_cont_handler_refines_stated;
      intros vs handler_vars conts' statics' hp' rd' _ Hvalues Hbind;
      apply equivalent_refines;
      [ | apply equivalent_sym ];
      exact
        (untaggings_at_head_equivalent params metadata vars vs handler_vars
           conts' statics' hp' rd' Hdistinct Hvalues Hbind
           (tagged_params params (hm_params_subkinds metadata))
           (bind_untaggings handler) (fun x Hx => Hx)).
Qed.

Theorem bind_untaggings_equivalent : bind_untaggings_equivalent_statement.
Proof. exact bind_untaggings_equivalent_aux. Qed.
