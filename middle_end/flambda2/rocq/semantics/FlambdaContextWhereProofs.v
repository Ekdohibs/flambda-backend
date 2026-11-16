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

(* Proofs of the statements of FlambdaContextWhereSpec.v. Nothing here
   needs review.

   Each statement is a lemma of FlambdaContextProofs.v, which are about
   any relations between handlers and between the bodies of code that
   give a refinement from the states that have the predicate, at the
   relations of FlambdaContextWhereSpec.v. The proof itself is there
   ([transfer]): a run under some environments is carried to environments
   whose closures and code are the rewritten ones. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaRangeSpec FlambdaRangeProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaContextWhereSpec.
Import ListNotations.

(* ======================================================================== *)
(* The relations of the Spec are ones that the lemmas ask for               *)
(* ======================================================================== *)

Section Relations.

Variable inv : state -> Prop.

(* From the target to the source. *)
Definition handler_refined_from (params : list variable)
    (kinds : list flambda_kind) (def_vars : env) (target source : expr)
    : Prop :=
  handler_refines_from inv params kinds def_vars source target.

Definition code_refined_from (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (target source : expr) : Prop :=
  code_refines_from inv params my_closure ret exn metadata source target.

Lemma handler_forwards (params : list variable)
    (kinds : list flambda_kind) (def_vars : env) (source target : expr)
    (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  handler_refines_from inv params kinds def_vars source target ->
  map kind_of_value vs = kinds ->
  bind_params params vs def_vars = Some handler_vars ->
  refines_from inv (State source handler_vars conts statics hp rd)
    (State target handler_vars conts statics hp rd).
Proof.
  intros Hrefines Hkinds Hbind.
  exact (Hrefines vs handler_vars conts statics hp rd_ Hkinds Hbind).
Qed.

Lemma handler_backwards (params : list variable)
    (kinds : list flambda_kind) (def_vars : env) (target source : expr)
    (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  handler_refined_from params kinds def_vars target source ->
  map kind_of_value vs = kinds ->
  bind_params params vs def_vars = Some handler_vars ->
  refines_from inv (State source handler_vars conts statics hp rd)
    (State target handler_vars conts statics hp rd).
Proof.
  intros Hrefines Hkinds Hbind.
  exact (Hrefines vs handler_vars conts statics hp rd_ Hkinds Hbind).
Qed.

(* The lemmas of FlambdaContextProofs.v give, of the callee, that it is
   a closure of the code id that the bodies are related at. These two
   relations do not depend on the code id, and the lemmas do not use
   that. *)
Lemma code_forwards (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (source target : expr) (vs : list value)
    (callee : value) (callee_vars : env) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record)
      (slots : list (value_slot * value)) :
  code_refines_from inv params my_closure ret exn metadata source target ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  map kind_of_value vs = unarize (cm_params_arity metadata) ->
  kind_of_value callee = Value ->
  closure_of statics callee = Some (id, slots) ->
  refines_from inv
    (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp rd)
    (State target callee_vars [(ret, return_to); (exn, exn_to)] statics hp rd).
Proof.
  intros Hrefines Hbind Hkinds Hvalue _.
  exact
    (Hrefines vs callee callee_vars return_to exn_to statics hp rd_ Hbind Hkinds
       Hvalue).
Qed.

Lemma code_backwards (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (target source : expr) (vs : list value)
    (callee : value) (callee_vars : env) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record)
      (slots : list (value_slot * value)) :
  code_refined_from params my_closure ret exn metadata target source ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  map kind_of_value vs = unarize (cm_params_arity metadata) ->
  kind_of_value callee = Value ->
  closure_of statics callee = Some (id, slots) ->
  refines_from inv
    (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp rd)
    (State target callee_vars [(ret, return_to); (exn, exn_to)] statics hp rd).
Proof.
  intros Hrefines Hbind Hkinds Hvalue _.
  exact
    (Hrefines vs callee callee_vars return_to exn_to statics hp rd_ Hbind Hkinds
       Hvalue).
Qed.

End Relations.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem let_cont_handler_refines_from
  : let_cont_handler_refines_from_statement.
Proof.
  intros inv k params metadata handler handler' body vars conts statics hp rd
    Hstep Hinv Hinv' Hhandler.
  exact
    (FlambdaContextProofs.let_cont_handler_refines_from inv Hstep _ _ _ _
       (handler_forwards inv) (handler_backwards inv) (code_forwards inv)
       (code_backwards inv) k params metadata handler handler' body vars
       conts statics hp rd_ Hinv Hinv' Hhandler Hhandler).
Qed.

Theorem let_cont_recursive_handler_refines_from
  : let_cont_recursive_handler_refines_from_statement.
Proof.
  intros inv before k params metadata handler handler' after body vars conts
    statics hp rd Hstep Hinv Hinv' Hbinds Hhandler.
  apply
    (let_cont_recursive_handlers_refine_from inv Hstep _ _ _ _
       (handler_forwards inv) (handler_backwards inv) (code_forwards inv)
       (code_backwards inv));
    [ exact Hinv | exact Hinv'
    | unfold handlers_bind_statics; rewrite !existsb_app; simpl;
      rewrite Hbinds; reflexivity
    | | ];
    (apply Forall2_app; [ apply handlers_related_refl | ]);
    (apply Forall2_cons; [ | apply handlers_related_refl ]);
    (split; [ reflexivity | ]); apply Handler_related; exact Hhandler.
Qed.

Theorem let_static_code_refines_from
  : let_static_code_refines_from_statement.
Proof.
  intros inv before id params my_closure ret exn metadata code_body
    code_body' after body vars conts statics hp rd Hstep Hinv Hinv' Hbinds
      Hcode.
  exact
    (FlambdaContextProofs.let_static_code_refines_from inv Hstep _ _ _ _
       (handler_forwards inv) (handler_backwards inv) (code_forwards inv)
       (code_backwards inv) before id params my_closure ret exn metadata
       code_body code_body' after body vars conts statics hp rd_ Hinv Hinv'
         Hbinds
       Hcode Hcode).
Qed.

Theorem let_static_code_refines_where
  : let_static_code_refines_where_statement.
Proof.
  intros ok before id params my_closure ret exn metadata code_body
    code_body' after body vars conts statics hp rd Hextend Hok Hbinds Hcode.
  exact
    (FlambdaContextProofs.let_static_code_refines_where ok Hextend before id
       params my_closure ret exn metadata code_body code_body' after body
       vars conts statics hp rd_ Hok Hbinds Hcode).
Qed.

(* ======================================================================== *)
(* The premises can be met                                                  *)
(* ======================================================================== *)

Theorem handler_refines_from_always : handler_refines_from_always_statement.
Proof.
  intros params kinds def_vars source target. split.
  - intros Hrefines vs handler_vars conts statics hp rd Hkinds Hbind.
    exact
      (proj2 (Hrefines vs handler_vars conts statics hp rd_ Hkinds Hbind) I).
  - intros Hrefines vs handler_vars conts statics hp rd Hkinds Hbind.
    split; [ split; intros _; exact I | intros _ ].
    exact (Hrefines vs handler_vars conts statics hp rd_ Hkinds Hbind).
Qed.

Theorem code_refines_from_always : code_refines_from_always_statement.
Proof.
  intros params my_closure ret exn metadata source target Hrefines vs callee
    callee_vars return_to exn_to statics hp rd _ _ _.
  split; [ split; intros _; exact I | intros _; apply Hrefines ].
Qed.

Theorem handler_refines_from_in_range
  : handler_refines_from_in_range_statement.
Proof.
  intros params kinds def_vars source target Hsource Htarget Hrefines vs
    handler_vars conts statics hp rd Hkinds Hbind.
  split;
    [ | exact (Hrefines vs handler_vars conts statics hp rd_ Hkinds Hbind) ].
  unfold state_in_range.
  cbn [state_expr state_vars state_conts state_statics].
  rewrite Hsource, Htarget. reflexivity.
Qed.

Theorem let_cont_handler_refines_in_range
  : let_cont_handler_refines_in_range_statement.
Proof.
  intros k params metadata handler handler' body vars conts statics hp rd Hinv
    Htarget Hrefines.
  pose proof Hinv as (Hexpr & Hvars & Hconts & Hstatics).
  cbn [state_expr state_vars state_conts state_statics] in Hexpr, Hvars,
    Hconts, Hstatics.
  simpl in Hexpr. apply andb_true_iff in Hexpr as [Hsource Hbody].
  apply (let_cont_handler_refines_from state_in_range);
    [ exact step_in_range | exact Hinv | | ].
  - split; [ | split; [ exact Hvars | split; [ exact Hconts | ] ] ];
      cbn [state_expr state_vars state_conts state_statics];
      [ | exact Hstatics ].
    simpl. rewrite Htarget, Hbody. reflexivity.
  - apply handler_refines_from_in_range; assumption.
Qed.

(* ======================================================================== *)
(* The body of code, from the states in which the callee has its code       *)
(* ======================================================================== *)

Section Called.

Variable inv : state -> Prop.

(* From the target to the source. *)
Definition code_refined_from_called (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (target source : expr) : Prop :=
  code_refines_from_called inv id params my_closure ret exn metadata source
    target.

(* The lemmas of FlambdaContextProofs.v give that the callee is a
   closure of the code id that the bodies are related at, which is what
   the premise asks for. *)
Lemma code_forwards_called (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (source target : expr) (vs : list value)
    (callee : value) (callee_vars : env) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record)
      (slots : list (value_slot * value)) :
  code_refines_from_called inv id params my_closure ret exn metadata source
    target ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  map kind_of_value vs = unarize (cm_params_arity metadata) ->
  kind_of_value callee = Value ->
  closure_of statics callee = Some (id, slots) ->
  refines_from inv
    (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp rd)
    (State target callee_vars [(ret, return_to); (exn, exn_to)] statics hp rd).
Proof.
  intros Hrefines Hbind Hkinds Hvalue Hclosure.
  exact
    (Hrefines vs callee callee_vars return_to exn_to statics hp rd_ slots Hbind
       Hkinds Hvalue Hclosure).
Qed.

Lemma code_backwards_called (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (target source : expr) (vs : list value)
    (callee : value) (callee_vars : env) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record)
      (slots : list (value_slot * value)) :
  code_refined_from_called id params my_closure ret exn metadata target
    source ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  map kind_of_value vs = unarize (cm_params_arity metadata) ->
  kind_of_value callee = Value ->
  closure_of statics callee = Some (id, slots) ->
  refines_from inv
    (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp rd)
    (State target callee_vars [(ret, return_to); (exn, exn_to)] statics hp rd).
Proof.
  intros Hrefines Hbind Hkinds Hvalue Hclosure.
  exact
    (Hrefines vs callee callee_vars return_to exn_to statics hp rd_ slots Hbind
       Hkinds Hvalue Hclosure).
Qed.

End Called.

Theorem let_static_code_refines_from_called
  : let_static_code_refines_from_called_statement.
Proof.
  intros inv before id params my_closure ret exn metadata code_body
    code_body' after body vars conts statics hp rd Hstep Hinv Hinv' Hbinds
      Hcode.
  exact
    (FlambdaContextProofs.let_static_code_refines_from inv Hstep _ _ _ _
       (handler_forwards inv) (handler_backwards inv)
       (code_forwards_called inv) (code_backwards_called inv) before id
       params my_closure ret exn metadata code_body code_body' after body
       vars conts statics hp rd_ Hinv Hinv' Hbinds Hcode Hcode).
Qed.

(* Two bodies that have the premise of [code_refines_from] have that of
   [code_refines_from_called], for every code id: the second is asked of
   fewer states. *)
Lemma code_refines_from_called_of_from (inv : state -> Prop) (id : code_id)
    (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (source target : expr) :
  code_refines_from inv params my_closure ret exn metadata source target ->
  code_refines_from_called inv id params my_closure ret exn metadata source
    target.
Proof.
  intros Hrefines vs callee callee_vars return_to exn_to statics hp rd
    value_slots Hbind Hkinds Hvalue _.
  exact
    (Hrefines vs callee callee_vars return_to exn_to statics hp rd_ Hbind Hkinds
       Hvalue).
Qed.

(* So [let_static_code_refines_from_called_statement] gives
   [let_static_code_refines_from_statement]. *)
Lemma let_static_code_refines_from_of_called :
  let_static_code_refines_from_called_statement ->
  let_static_code_refines_from_statement.
Proof.
  intros Hcalled inv before id params my_closure ret exn metadata code_body
    code_body' after body vars conts statics hp rd Hstep Hinv Hinv' Hbinds
    Hbody.
  exact
    (Hcalled inv before id params my_closure ret exn metadata code_body
       code_body' after body vars conts statics hp rd Hstep Hinv Hinv' Hbinds
       (code_refines_from_called_of_from inv id params my_closure ret exn
          metadata code_body code_body' Hbody)).
Qed.
