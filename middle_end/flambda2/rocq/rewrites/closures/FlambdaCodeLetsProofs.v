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

(* Proofs of the statements of FlambdaCodeLetsSpec.v. Nothing here needs
   review.

   The body of code is rewritten by
   FlambdaLoopLetsSpec.remove_unused_defined_lets_in_loops, which is an
   equivalence under variables bound to values of their kinds
   (FlambdaLoopLetsProofs.remove_unused_defined_lets_in_loops_equivalent_bound),
   and a call binds the variables of [code_in_scope] so
   ([code_in_scope_hold]). So the new body refines the old one where it
   runs, and the old one the new, and a group of static bindings may have
   the bodies of its code replaced, one piece of code at a time
   (FlambdaContextProofs.let_static_code_refines_where). The rest is the
   induction of FlambdaNormaliseProofs.rewrite_code_equivalent. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaLoopLetsSpec FlambdaLoopLetsProofs FlambdaCodeLetsSpec.
Import ListNotations.

Fixpoint remove_unused_defined_lets_in_code_binds_statics (e : expr) :
  binds_statics (remove_unused_defined_lets_in_code e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply remove_unused_defined_lets_in_code_binds_statics.
  - rewrite remove_unused_defined_lets_in_code_binds_statics. f_equal.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !remove_unused_defined_lets_in_code_binds_statics. reflexivity.
  - rewrite remove_unused_defined_lets_in_code_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite remove_unused_defined_lets_in_code_binds_statics, IH.
    reflexivity.
Qed.

(* What a call binds has the kinds of [code_in_scope]. *)
Lemma code_in_scope_hold (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (vs : list value) (callee : value)
    (callee_vars : env) :
  map kind_of_value vs = unarize (cm_params_arity metadata) ->
  kind_of_value callee = Value ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  kinds_hold (code_in_scope params my_closure metadata) callee_vars.
Proof.
  intros Hkinds Hvalue Hbind. unfold code_in_scope.
  rewrite <- Hkinds, <- Hvalue.
  apply
    (params_in_scope_hold params vs _ [(my_closure, callee)] callee_vars);
    [ | exact Hbind ].
  apply (kinds_hold_cons [] [] my_closure callee).
  intros x kind [=].
Qed.

(* The body of code and the body as rewritten refine each other where
   the code runs. *)
Lemma code_body_refines (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (body : expr) :
  code_refines_where always params my_closure metadata body
    (remove_unused_defined_lets_in_loops
       (code_in_scope params my_closure metadata) body)
  ∧ code_refines_where always params my_closure metadata
      (remove_unused_defined_lets_in_loops
         (code_in_scope params my_closure metadata) body)
      body.
Proof.
  split; intros vs callee callee_vars conts statics hp rd _ Hkinds Hvalue Hbind;
    apply equivalent_refines; [ | apply equivalent_sym ];
    apply remove_unused_defined_lets_in_loops_equivalent_bound,
      (code_in_scope_hold params my_closure metadata vs callee callee_vars
         Hkinds Hvalue Hbind).
Qed.

(* A group of static bindings with the bodies of its code rewritten, one
   piece of code at a time: [done] is what has been rewritten. *)
Lemma bindings_in_code_equivalent (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (todo : list static_binding) :
  ∀ done : list static_binding,
    equivalent (State (Let_static (done ++ todo) body) vars conts statics hp rd)
      (State
         (Let_static
            (done ++ map remove_unused_defined_lets_in_binding todo) body)
         vars conts statics hp rd).
Proof.
  induction todo as [ | binding todo IH]; intros done; simpl;
    [ apply equivalent_refl | ].
  apply (equivalent_trans _
    (State
       (Let_static
          (done ++ remove_unused_defined_lets_in_binding binding :: todo)
          body)
       vars conts statics hp rd)).
  - destruct binding
      as [id [params my_closure ret exn metadata code_body] | id
         | sym const];
      simpl; try apply equivalent_refl.
    destruct (code_body_refines params my_closure metadata code_body)
      as [Hthere Hback].
    apply refines_antisymmetric;
      apply (let_static_code_refines_where always always_extend);
      first
        [ exact I | exact Hthere | exact Hback
        | apply remove_unused_defined_lets_in_loops_binds_statics
        | symmetry; apply remove_unused_defined_lets_in_loops_binds_statics ].
  - specialize
      (IH (done ++ [remove_unused_defined_lets_in_binding binding])).
    rewrite <- !app_assoc in IH. exact IH.
Qed.

Fixpoint remove_unused_defined_lets_in_code_equivalent_aux (e : expr) :
  expr_equivalent e (remove_unused_defined_lets_in_code e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; intros vars conts statics hp rd; try apply equivalent_refl.
  - apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply remove_unused_defined_lets_in_code_equivalent_aux.
  - (* down the body, then the code of the group *)
    apply (equivalent_trans _
      (State (Let_static bindings (remove_unused_defined_lets_in_code body))
         vars conts statics hp rd)).
    + apply let_static_equivalent_step.
      destruct (eval_static_bindings vars bindings statics) as [extended | ];
        [ | exact I ].
      apply remove_unused_defined_lets_in_code_equivalent_aux.
    + exact
      (bindings_in_code_equivalent _ vars conts statics _ rd_ bindings []).
  - apply (equivalent_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (remove_unused_defined_lets_in_code handler))
            body)
         vars conts statics hp rd)).
    + apply
        (let_cont_handler_equivalent_bound [] k params metadata handler _
           body);
        [ | intros x kind [=] ].
      intros vars' conts' statics' hp' rd' _.
      apply remove_unused_defined_lets_in_code_equivalent_aux.
    + apply let_cont_non_recursive_equivalent_step.
      apply remove_unused_defined_lets_in_code_equivalent_aux.
  - apply (equivalent_trans _
      (State
         (Let_cont_recursive
            (map
               (fun '(k, h) =>
                  (k, remove_unused_defined_lets_in_code_handler h))
               handlers)
            body)
         vars conts statics hp rd)).
    + apply (let_cont_recursive_handlers_equivalent_bound [] handlers _ body);
        [ | | intros x kind [=] ].
      * induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ apply Forall2_nil | ].
        apply Forall2_cons; [ | exact IH ].
        split; [ reflexivity | ]. simpl.
        exists params, metadata, handler,
          (remove_unused_defined_lets_in_code handler).
        split; [ reflexivity | ]. split; [ reflexivity | ].
        intros vars' conts' statics' hp' rd' _.
        apply remove_unused_defined_lets_in_code_equivalent_aux.
      * unfold handlers_bind_statics.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ reflexivity | ].
        rewrite remove_unused_defined_lets_in_code_binds_statics, IH.
        reflexivity.
    + apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply remove_unused_defined_lets_in_code_equivalent_aux.
Qed.

Theorem remove_unused_defined_lets_in_code_equivalent :
  remove_unused_defined_lets_in_code_equivalent_statement.
Proof. exact remove_unused_defined_lets_in_code_equivalent_aux. Qed.
