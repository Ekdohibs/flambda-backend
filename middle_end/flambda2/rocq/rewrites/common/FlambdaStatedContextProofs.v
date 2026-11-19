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

(* Equivalence under what is known of the variables in scope, and its
   laws through the binders. There is no statement here, and nothing
   here needs review.

   A rewrite that reads what is known of the variables bound around a
   node of a term (FlambdaStatedScopeSpec.stated_in_scope) gives, at
   that node, a term that is equivalent under the environments of the
   machine in which what is known holds
   (FlambdaStatedScopeSpec.stated_hold), and not under all.
   [expr_equivalent_stated] is that relation. With nothing known it is
   FlambdaSemanticsSpec.expr_equivalent ([expr_equivalent_of_stated]).

   The laws take it through a binder, in step with what the rewrites
   add to what is known as they go down a term:
   - under a [Let], with what its defining expression makes
     ([let_equivalent_stated]);
   - under a [Let_static] ([let_static_equivalent_stated]): the static
     environment grows, and what a value is of it is still of;
   - in the body of a [Let_cont], with nothing added
     ([let_cont_body_equivalent_stated],
     [let_cont_recursive_body_equivalent_stated]);
   - in the handler of a [Let_cont], with what the handler states of
     its parameters ([let_cont_handler_equivalent_stated], and
     [let_cont_recursive_handler_equivalent_stated] for one handler of
     a recursive [Let_cont]).
   They are what FlambdaNormaliseProofs.expr_equivalent_bound and its
   laws are for the kinds of the variables in scope.

   The law of a handler is not the one of the kinds with another
   invariant. What is known of a variable is tested under a static
   environment (a symbol is of a subkind by what it is bound to), and
   a handler is entered under a static environment that may have grown
   since its [Let_cont] ran. So the handlers are compared where they
   are entered under a static environment that extends the one of the
   [Let_cont] ([handler_refines_stated_where], which is
   FlambdaContextProofs.handler_refines_stated with that added), and
   what was known at the [Let_cont] is carried to it
   (FlambdaStatedScopeProofs.stated_hold_extend). The values that
   enter are of what the handler states, under that static
   environment: the machine has just tested them.

   Not here: the body of code that is entered. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSubkindSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaContextProofs FlambdaStatedScopeSpec FlambdaStatedScopeProofs.
Import ListNotations.

(* Equivalence under the environments in which what is known of the
   variables in scope holds. *)
Definition expr_equivalent_stated (S : stated_in_scope) (e e' : expr)
    : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    stated_hold statics S vars ->
    equivalent (State e vars conts statics hp rd)
      (State e' vars conts statics hp rd).

Lemma expr_equivalent_stated_refl (S : stated_in_scope) (e : expr) :
  expr_equivalent_stated S e e.
Proof. intros vars conts statics hp rd _. apply equivalent_refl. Qed.

Lemma expr_equivalent_stated_trans (S : stated_in_scope) (e e' e'' : expr) :
  expr_equivalent_stated S e e' ->
  expr_equivalent_stated S e' e'' ->
  expr_equivalent_stated S e e''.
Proof.
  intros H H' vars conts statics hp rd Hhold.
  exact (equivalent_trans _ _ _ (H vars conts statics hp rd_ Hhold)
    (H' vars conts statics hp rd_ Hhold)).
Qed.

(* An equivalence under all environments is one under these. *)
Lemma expr_equivalent_stated_of (S : stated_in_scope) (e e' : expr) :
  expr_equivalent e e' -> expr_equivalent_stated S e e'.
Proof. intros H vars conts statics hp rd _. apply H. Qed.

(* With nothing known, all environments. *)
Lemma expr_equivalent_of_stated (e e' : expr) :
  expr_equivalent_stated [] e e' -> expr_equivalent e e'.
Proof.
  intros H vars conts statics hp rd. apply H, stated_hold_nil.
Qed.

(* ======================================================================== *)
(* The binders that the machine goes under                                  *)
(* ======================================================================== *)

Lemma let_equivalent_stated (S : stated_in_scope) (x : variable)
    (kind : flambda_kind) (d : named) (body body' : expr) :
  expr_equivalent_stated ((x, stated_of_named S d) :: S) body body' ->
  expr_equivalent_stated S (Let x kind d body) (Let x kind d body').
Proof.
  intros Hbody vars conts statics hp rd Hhold.
  apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' Hexec _ ].
  apply Hbody, stated_hold_cons; [ exact Hhold | ].
  exact (stated_of_named_sound statics hp hp' rd rd' S vars d v Hhold Hexec).
Qed.

Lemma let_static_equivalent_stated (S : stated_in_scope)
    (bindings : list static_binding) (body body' : expr) :
  expr_equivalent_stated S body body' ->
  expr_equivalent_stated S (Let_static bindings body)
    (Let_static bindings body').
Proof.
  intros Hbody vars conts statics hp rd Hhold.
  apply let_static_equivalent_step.
  destruct (eval_static_bindings vars bindings statics) as [extended | ]
    eqn:Hbindings; [ | exact I ].
  apply Hbody.
  exact
    (stated_hold_extend statics extended S vars
       (eval_static_bindings_extend vars _ _ _ Hbindings) Hhold).
Qed.

Lemma let_cont_body_equivalent_stated (S : stated_in_scope)
    (k : continuation) (h : continuation_handler) (body body' : expr) :
  expr_equivalent_stated S body body' ->
  expr_equivalent_stated S (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h body').
Proof.
  intros Hbody vars conts statics hp rd Hhold.
  apply let_cont_non_recursive_equivalent_step, Hbody, Hhold.
Qed.

Lemma let_cont_recursive_body_equivalent_stated (S : stated_in_scope)
    (handlers : list (continuation * continuation_handler))
    (body body' : expr) :
  expr_equivalent_stated S body body' ->
  expr_equivalent_stated S (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers body').
Proof.
  intros Hbody vars conts statics hp rd Hhold.
  apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
  apply Hbody, Hhold.
Qed.

(* ======================================================================== *)
(* The handlers                                                             *)
(* ======================================================================== *)

(* A handler refines another where it is entered under a static
   environment of which [ok] holds: as
   FlambdaContextProofs.handler_refines_stated, which is this at every
   static environment. *)
Definition handler_refines_stated_where (ok : static_env -> Prop)
    (params : list variable) (metadata : handler_metadata) (def_vars : env)
    (source target : expr) : Prop :=
  ∀ (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    ok statics ->
    map kind_of_value vs = hm_params_kinds metadata ->
    values_of_subkinds statics vs (hm_params_subkinds metadata) = true ->
    bind_params params vs def_vars = Some handler_vars ->
    refines (State source handler_vars conts statics hp rd)
      (State target handler_vars conts statics hp rd).

Section Where.

Variable ok : static_env -> Prop.

Hypothesis ok_extend :
  ∀ statics statics' : static_env,
    ok statics -> statics_extend statics statics' -> ok statics'.

(* The relation and the same read the other way are the two that
   FlambdaContextProofs.let_cont_handler_refines_from wants, from the
   states whose static environment has [ok]. *)
Definition refined_stated_where (params : list variable)
    (metadata : handler_metadata) (def_vars : env) (target source : expr)
    : Prop :=
  handler_refines_stated_where ok params metadata def_vars source target.

Lemma stated_where_forwards (params : list variable)
    (metadata : handler_metadata) (def_vars : env) (source target : expr)
    (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  handler_refines_stated_where ok params metadata def_vars source target ->
  map kind_of_value vs = hm_params_kinds metadata ->
  values_of_subkinds statics vs (hm_params_subkinds metadata) = true ->
  bind_params params vs def_vars = Some handler_vars ->
  refines_from (statics_ok ok) (State source handler_vars conts statics hp rd)
    (State target handler_vars conts statics hp rd).
Proof.
  intros Hrefines Hkinds Hsubkinds Hbind. apply statics_ok_refines_from.
  intros Hok.
  exact
    (Hrefines vs handler_vars conts statics hp rd_ Hok Hkinds Hsubkinds
       Hbind).
Qed.

Lemma stated_where_backwards (params : list variable)
    (metadata : handler_metadata) (def_vars : env) (target source : expr)
    (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  refined_stated_where params metadata def_vars target source ->
  map kind_of_value vs = hm_params_kinds metadata ->
  values_of_subkinds statics vs (hm_params_subkinds metadata) = true ->
  bind_params params vs def_vars = Some handler_vars ->
  refines_from (statics_ok ok) (State source handler_vars conts statics hp rd)
    (State target handler_vars conts statics hp rd).
Proof. apply stated_where_forwards. Qed.

(* A [Let_cont] refines the same with the handler replaced by one that
   refines it where it is entered. The two handlers have the same
   parameters and the same metadata, so the law may be taken once each
   way. *)
Lemma let_cont_handler_refines_stated_where (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  ok statics ->
  handler_refines_stated_where ok params metadata vars handler handler' ->
  refines
    (State
       (Let_cont_non_recursive k (Handler params metadata handler) body)
       vars conts statics hp rd)
    (State
       (Let_cont_non_recursive k (Handler params metadata handler') body)
       vars conts statics hp rd).
Proof.
  intros Hok Hhandler.
  refine
    (let_cont_handler_refines_from (statics_ok ok)
       (statics_ok_step ok ok_extend) (handler_refines_stated_where ok)
       refined_stated_where (fun _ _ _ _ _ _ _ _ => False)
       (fun _ _ _ _ _ _ _ _ => False) stated_where_forwards
       stated_where_backwards _ _ k params metadata handler handler' body
       vars conts statics hp rd Hok Hok Hhandler Hhandler);
    intros; exfalso; assumption.
Qed.

(* The same for one handler of a recursive [Let_cont]. *)
Lemma let_cont_recursive_handler_refines_stated_where
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr)
    (after : list (continuation * continuation_handler)) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record) :
  ok statics ->
  binds_statics handler = binds_statics handler' ->
  handler_refines_stated_where ok params metadata vars handler handler' ->
  refines
    (State
       (Let_cont_recursive
          (before ++ (k, Handler params metadata handler) :: after) body)
       vars conts statics hp rd)
    (State
       (Let_cont_recursive
          (before ++ (k, Handler params metadata handler') :: after) body)
       vars conts statics hp rd).
Proof.
  intros Hok Hbinds Hhandler.
  refine
    (let_cont_recursive_handlers_refine_from (statics_ok ok)
       (statics_ok_step ok ok_extend) (handler_refines_stated_where ok)
       refined_stated_where (fun _ _ _ _ _ _ _ _ => False)
       (fun _ _ _ _ _ _ _ _ => False) stated_where_forwards
       stated_where_backwards _ _ _ _ body vars conts statics hp rd Hok Hok
       _ _ _);
    try (intros; exfalso; assumption).
  - unfold handlers_bind_statics. rewrite !existsb_app. simpl.
    rewrite Hbinds. reflexivity.
  - apply Forall2_app; [ apply handlers_related_refl | ].
    apply Forall2_cons; [ | apply handlers_related_refl ].
    split; [ reflexivity | ]. apply Handler_related. exact Hhandler.
  - apply Forall2_app; [ apply handlers_related_refl | ].
    apply Forall2_cons; [ | apply handlers_related_refl ].
    split; [ reflexivity | ]. apply Handler_related. exact Hhandler.
Qed.

End Where.

(* What was known where a [Let_cont] ran, with what the handler states
   of its parameters, holds where the handler is entered. *)
Lemma params_stated_entered (S : stated_in_scope) (params : list variable)
    (metadata : handler_metadata) (vars : env) (statics statics' : static_env)
    (vs : list value) (handler_vars : env) :
  stated_hold statics S vars ->
  statics_extend statics statics' ->
  values_of_subkinds statics' vs (hm_params_subkinds metadata) = true ->
  bind_params params vs vars = Some handler_vars ->
  stated_hold statics'
    (params_stated params (hm_params_subkinds metadata) S) handler_vars.
Proof.
  intros Hhold Hext Hvs Hbind.
  exact
    (params_stated_bound statics' params vs _ S vars handler_vars
       (stated_hold_extend statics statics' S vars Hext Hhold) Hvs Hbind).
Qed.

Lemma extend_from (statics : static_env) :
  ∀ statics' statics'' : static_env,
    statics_extend statics statics' ->
    statics_extend statics' statics'' ->
    statics_extend statics statics''.
Proof. exact (statics_extend_trans statics). Qed.

Lemma let_cont_handler_equivalent_stated (S : stated_in_scope)
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr) :
  expr_equivalent_stated
    (params_stated params (hm_params_subkinds metadata) S) handler
    handler' ->
  expr_equivalent_stated S
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  intros Hhandler vars conts statics hp rd Hhold.
  apply refines_antisymmetric;
    apply
      (let_cont_handler_refines_stated_where (statics_extend statics)
         (extend_from statics));
    try apply statics_extend_refl;
    intros vs handler_vars conts' statics' hp' rd' Hext Hkinds Hvs Hbind;
    [ apply equivalent_refines | apply equivalent_refines, equivalent_sym ];
    apply Hhandler;
    exact
      (params_stated_entered S params metadata vars statics statics' vs
         handler_vars Hhold Hext Hvs Hbind).
Qed.

(* The handlers of a recursive [Let_cont] are bound, as that of a
   non-recursive one, under the variables of the [Let_cont]. *)
Lemma let_cont_recursive_handler_equivalent_stated (S : stated_in_scope)
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr)
    (after : list (continuation * continuation_handler)) (body : expr) :
  binds_statics handler = binds_statics handler' ->
  expr_equivalent_stated
    (params_stated params (hm_params_subkinds metadata) S) handler
    handler' ->
  expr_equivalent_stated S
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler) :: after) body)
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler') :: after) body).
Proof.
  intros Hbinds Hhandler vars conts statics hp rd Hhold.
  apply refines_antisymmetric;
    apply
      (let_cont_recursive_handler_refines_stated_where
         (statics_extend statics) (extend_from statics));
    try apply statics_extend_refl; try congruence;
    intros vs handler_vars conts' statics' hp' rd' Hext Hkinds Hvs Hbind;
    [ apply equivalent_refines | apply equivalent_refines, equivalent_sym ];
    apply Hhandler;
    exact
      (params_stated_entered S params metadata vars statics statics' vs
         handler_vars Hhold Hext Hvs Hbind).
Qed.
