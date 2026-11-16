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

(* Proofs of the statements of FlambdaStaticSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaStaticSpec.
Import ListNotations.

(* Binding a group is binding its first part, then the rest. *)
Lemma eval_static_bindings_app (vars : env)
    (bindings bindings' : list static_binding) :
  ∀ statics : static_env,
    eval_static_bindings vars (bindings ++ bindings') statics
    = match eval_static_bindings vars bindings statics with
      | Some statics' => eval_static_bindings vars bindings' statics'
      | None => None
      end.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics;
    [ reflexivity | ].
  destruct binding as [id c | id | sym const]; simpl.
  - destruct (lookup id (static_code statics)); [ reflexivity | apply IH ].
  - apply IH.
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    destruct (eval_static_const vars const); simpl;
      [ apply IH | reflexivity ].
Qed.

Theorem let_static_split : let_static_split_statement.
Proof.
  intros bindings bindings' body vars conts statics hp rd.
  destruct (eval_static_bindings vars bindings statics) as [statics' | ]
    eqn:Hbindings.
  - (* the first part is bound: the second step of the right-hand side is
       the step of the left-hand side *)
    apply equivalent_sym.
    eapply equivalent_trans;
      [ apply step_equivalent; rewrite step_let_static, Hbindings;
        reflexivity
      | ].
    apply step_eq_equivalent.
    rewrite !step_let_static, eval_static_bindings_app, Hbindings.
    reflexivity.
  - apply step_eq_equivalent.
    rewrite !step_let_static, eval_static_bindings_app, Hbindings.
    reflexivity.
Qed.

(* Both sides bind the symbol to the closure, if the contents of the slot
   have a value, and then bind the variable to that value. *)
Theorem project_static_closure : project_static_closure_statement.
Proof.
  intros s code slot contents x kind body vars conts statics hp rd.
  apply let_static_equivalent_step. simpl.
  destruct (lookup s (static_values statics)); [ exact I | ].
  destruct (eval_simple vars contents) as [v | ] eqn:Hcontents; simpl;
    [ | exact I ].
  apply let_equivalent_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | | intros v' _ _;
     apply equivalent_refl ].
  simpl. rewrite !Pos.eqb_refl. simpl. rewrite ?Pos.eqb_refl, ?Hcontents.
  reflexivity.
Qed.

Theorem let_static_code_refines : let_static_code_refines_statement.
Proof.
  intros before id params my_closure ret exn metadata code_body code_body'
    after body Hbinds Hcode.
  exact (plug_refines
    (Let_static_code before id params my_closure ret exn metadata Hole after
       body)
    code_body code_body' (fun _ => Hbinds) Hcode).
Qed.
