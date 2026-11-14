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

(* Proofs of the statements of FlambdaLoopLetsSpec.v. Nothing here needs
   review.

   The induction is that of
   FlambdaNormaliseProofs.remove_unused_defined_lets_equivalent_bound,
   under variables that are bound to values of their kinds. At a
   recursive [Let_cont] the handlers are replaced all at once
   (FlambdaContextProofs.let_cont_recursive_handlers_refine_where): each
   new handler is equivalent to the old one under the variables of the
   closure with the parameters bound to values of the kinds of the
   handler, as those bind the variables that were in scope and the
   parameters (FlambdaNormaliseProofs.params_in_scope_hold). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaLoopLetsSpec.
Import ListNotations.

Fixpoint remove_unused_defined_lets_in_loops_binds_statics (e : expr) :
  ∀ kinds : kinds_in_scope,
    binds_statics (remove_unused_defined_lets_in_loops kinds e)
    = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros kinds; simpl; try reflexivity.
  - destruct
      (named_defined kinds kind d
       && negb
            (var_free_b x
               (remove_unused_defined_lets_in_loops ((x, kind) :: kinds)
                  body)));
      simpl; apply remove_unused_defined_lets_in_loops_binds_statics.
  - rewrite remove_unused_defined_lets_in_loops_binds_statics. reflexivity.
  - rewrite !remove_unused_defined_lets_in_loops_binds_statics. reflexivity.
  - rewrite remove_unused_defined_lets_in_loops_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
      [ reflexivity | ].
    rewrite remove_unused_defined_lets_in_loops_binds_statics, IH.
    reflexivity.
Qed.

(* Handlers that are equivalent, one with one, where the variables given
   and their parameters are bound to values of their kinds. *)
Definition handlers_equivalent_bound (kinds : kinds_in_scope)
    (handlers handlers' : list (continuation * continuation_handler)) : Prop :=
  Forall2
    (fun kh kh' =>
       fst kh = fst kh'
       ∧ ∃ (params : list variable) (metadata : handler_metadata)
           (handler handler' : expr),
           snd kh = Handler params metadata handler
           ∧ snd kh' = Handler params metadata handler'
           ∧ expr_equivalent_bound
               (params_in_scope params (hm_params_kinds metadata) kinds)
               handler handler')
    handlers handlers'.

(* Such handlers refine each other where they run, in the four forms
   that the replacement of the handlers of a recursive [Let_cont] asks
   for. *)
Lemma handlers_equivalent_bound_related (kinds : kinds_in_scope) (vars : env)
    (handlers handlers' : list (continuation * continuation_handler)) :
  kinds_hold kinds vars ->
  handlers_equivalent_bound kinds handlers handlers' ->
  handlers_refine always vars handlers handlers'
  ∧ handlers_refine always vars handlers' handlers.
Proof.
  intros Hhold Hall. unfold handlers_refine.
  induction Hall
    as [ | [k h] [k' h'] handlers handlers'
           (Hk & params & metadata & handler & handler' & Hh & Hh' & Hequiv) _
           [[Hfwd Hbwd] [Hfwd' Hbwd']]].
  - repeat split; apply Forall2_nil.
  - simpl in Hk, Hh, Hh'. subst k' h h'.
    assert (Hruns :
      ∀ (values : list value) (handler_vars : env),
        map kind_of_value values = hm_params_kinds metadata ->
        bind_params params values vars = Some handler_vars ->
        kinds_hold (params_in_scope params (hm_params_kinds metadata) kinds)
          handler_vars).
    { intros values handler_vars <-.
      exact
        (params_in_scope_hold params values kinds vars handler_vars Hhold). }
    repeat split; (apply Forall2_cons; [ | assumption ]);
      (split; [ reflexivity | ]); apply Handler_related;
      intros values handler_vars conts statics hp rd _ Hkinds Hbind;
      apply equivalent_refines;
      first
        [ apply Hequiv, (Hruns values handler_vars Hkinds Hbind)
        | apply equivalent_sym, Hequiv,
            (Hruns values handler_vars Hkinds Hbind) ].
Qed.

Lemma let_cont_recursive_handlers_equivalent_bound (kinds : kinds_in_scope)
    (handlers handlers' : list (continuation * continuation_handler))
    (body : expr) :
  handlers_equivalent_bound kinds handlers handlers' ->
  handlers_bind_statics handlers = handlers_bind_statics handlers' ->
  expr_equivalent_bound kinds (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers' body).
Proof.
  intros Hall Hbinds vars conts statics hp rd Hhold.
  destruct
    (handlers_equivalent_bound_related kinds vars handlers handlers' Hhold
       Hall)
    as [[Hfwd Hbwd] [Hfwd' Hbwd']].
  apply refines_antisymmetric;
    apply (let_cont_recursive_handlers_refine_where always always_extend);
    first [ exact I | assumption | symmetry; assumption ].
Qed.

Fixpoint remove_unused_defined_lets_in_loops_equivalent_bound (e : expr) :
  ∀ kinds : kinds_in_scope,
    expr_equivalent_bound kinds e
      (remove_unused_defined_lets_in_loops kinds e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros kinds; simpl;
    try (intros vars conts statics hp rd _; apply equivalent_refl).
  - pose proof (let_equivalent_bound kinds x kind d _ _
      (remove_unused_defined_lets_in_loops_equivalent_bound body
         ((x, kind) :: kinds)))
      as Hbody.
    destruct
      (named_defined kinds kind d
       && negb
            (var_free_b x
               (remove_unused_defined_lets_in_loops ((x, kind) :: kinds)
                  body)))
      eqn:Hremove; [ | exact Hbody ].
    apply andb_true_iff in Hremove as [Hdefined Hunused].
    apply negb_true_iff in Hunused.
    apply (expr_equivalent_bound_trans kinds _ _ _ Hbody),
      unused_defined_let;
      [ exact Hdefined | ].
    intros Hx. apply var_free_b_sound in Hx. congruence.
  - intros vars conts statics hp rd Hhold.
    apply let_static_equivalent_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    apply (remove_unused_defined_lets_in_loops_equivalent_bound body kinds),
      Hhold.
  - apply (expr_equivalent_bound_trans kinds _
      (Let_cont_non_recursive k
         (Handler params metadata
            (remove_unused_defined_lets_in_loops
               (params_in_scope params (hm_params_kinds metadata) kinds)
               handler))
         body)).
    + apply let_cont_handler_equivalent_bound,
        remove_unused_defined_lets_in_loops_equivalent_bound.
    + intros vars conts statics hp rd Hhold.
      apply let_cont_non_recursive_equivalent_step.
      apply (remove_unused_defined_lets_in_loops_equivalent_bound body kinds),
        Hhold.
  - apply (expr_equivalent_bound_trans kinds _
      (Let_cont_recursive
         (map
            (fun '(k, h) =>
               (k, remove_unused_defined_lets_in_handler kinds h))
            handlers)
         body)).
    + apply let_cont_recursive_handlers_equivalent_bound.
      * induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ apply Forall2_nil | ].
        apply Forall2_cons; [ | exact IH ].
        split; [ reflexivity | ]. simpl.
        exists params, metadata, handler,
          (remove_unused_defined_lets_in_loops
             (params_in_scope params (hm_params_kinds metadata) kinds)
             handler).
        split; [ reflexivity | ]. split; [ reflexivity | ].
        apply remove_unused_defined_lets_in_loops_equivalent_bound.
      * unfold handlers_bind_statics.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ reflexivity | ].
        rewrite remove_unused_defined_lets_in_loops_binds_statics, IH.
        reflexivity.
    + intros vars conts statics hp rd Hhold.
      apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply (remove_unused_defined_lets_in_loops_equivalent_bound body kinds),
        Hhold.
Qed.

Theorem remove_unused_defined_lets_in_loops_equivalent :
  remove_unused_defined_lets_in_loops_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (remove_unused_defined_lets_in_loops_equivalent_bound e []).
  intros x kind [=].
Qed.
