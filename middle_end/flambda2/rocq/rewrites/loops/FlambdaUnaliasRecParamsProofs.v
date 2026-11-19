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

(* Proofs of the statement of FlambdaUnaliasRecParamsSpec.v. Nothing
   here needs review.

   One parameter of the one continuation of a recursive [Let_cont]
   first, then the pass. The one parameter is the site of
   FlambdaUnaliasParamsProofs.v, which is proved for a loop too: the
   closure of the continuation holds itself, and in its handler the
   applications give the parameter, which holds what the simple given
   from outside held where the [Let_cont] was run, which is of what
   the handler states of the parameter: what is known of the variables
   bound around the [Let_cont] says so, and holds where the [Let_cont]
   runs. The pass is then an induction on the term under what is known
   of the variables in scope (FlambdaStatedContextProofs.v). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaSimulationProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnusedParamsProofs
  FlambdaStatedScopeSpec FlambdaStatedContextProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaUnaliasRecParamsSpec.
From OxCamlFlambda2 Require FlambdaWellFormedProofs.
Import ListNotations.

(* ======================================================================== *)
(* The loop, of the terms                                                   *)
(* ======================================================================== *)

(* A parameter of the one continuation of a recursive [Let_cont] that
   every application outside the handler gives one simple, which no
   binder on the way binds and which is none of the other parameters
   of the handler, and that every application in the handler gives
   the parameter itself, is taken away and bound at the head of the
   handler: under the environments where the value of the simple is of
   what the handler states of the parameter. *)
Lemma loop_param_equivalent (i : nat) (k : continuation) (s : simple)
    (p : variable) (kind : flambda_kind) (params : list variable)
    (metadata : handler_metadata) (handler body : expr) :
  nth_error params i = Some p ->
  nth_error (hm_params_kinds metadata) i = Some kind ->
  simple_among s (drop_nth i params) = false ->
  among p (drop_nth i params) = false ->
  all_give_simple k i s true body = true ->
  all_give_simple k i (Var p) true handler = true ->
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    given_is_stated statics vars (hm_params_subkinds metadata) i s ->
    equivalent
      (State (Let_cont_recursive [(k, Handler params metadata handler)] body)
         vars conts statics hp rd)
      (State
         (Let_cont_recursive
            [(k,
              Handler (drop_nth i params)
                {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
                   hm_params_subkinds :=
                     drop_nth i (hm_params_subkinds metadata);
                   hm_is_exn_handler := false |}
                (Let p kind (Simple s) (drop_arg k i handler)))]
            (drop_arg k i body))
         vars conts statics hp rd).
Proof.
  intros Hparam Hkind Hs Hfresh Hgive Hturns vars conts statics hp rd
    Hstated.
  set (h' :=
    Handler (drop_nth i params)
      {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
         hm_params_subkinds :=
           drop_nth i (hm_params_subkinds metadata);
         hm_is_exn_handler := false |}
      (Let p kind (Simple s) (drop_arg k i handler))).
  assert (Hstatics :
    handlers_bind_statics [(k, h')]
    = handlers_bind_statics [(k, Handler params metadata handler)]).
  { unfold handlers_bind_statics, h'.
    cbn [existsb handler_binds_statics binds_statics].
    rewrite drop_arg_binds_statics. reflexivity. }
  destruct (handlers_bind_statics [(k, Handler params metadata handler)])
    eqn:Hbinds.
  - apply step_eq_equivalent.
    rewrite !step_let_cont_recursive, Hstatics, Hbinds. reflexivity.
  - assert (Hstep :
      step
        (State
           (Let_cont_recursive [(k, Handler params metadata handler)] body)
           vars conts statics hp rd)
      = Step
          (State body vars
             (site_conts true k params metadata handler vars conts) statics
             hp rd)).
    { rewrite step_let_cont_recursive, Hbinds. reflexivity. }
    apply (equivalent_trans _ _ _ (step_equivalent _ _ Hstep)).
    apply (equivalent_trans _
      (State (drop_arg k i body) vars
         (site_conts_binding true k i s p kind params metadata handler vars
            conts)
         statics hp rd));
      [ | apply equivalent_sym, step_equivalent;
          rewrite step_let_cont_recursive, Hstatics; reflexivity ].
    apply
      (simple_site_equivalent true k i s p kind params metadata handler vars
         conts statics Hparam Hkind Hs Hfresh Hstated (fun _ => Hturns));
      [ | apply FlambdaWellFormedProofs.statics_extend_refl ].
    apply (Rel_state_in_scope true k i s p kind params metadata handler vars
      conts s true);
      [ apply agree_refl | reflexivity | exact Hgive | ].
    exact
      (loop_conts_in_site true k i s p kind params metadata handler vars
         conts eq_refl).
Qed.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

Lemma unalias_rec_param_equivalent (S : stated_in_scope) (i : nat)
    (k : continuation) (h : continuation_handler) (body : expr)
    (h' : continuation_handler) (body' : expr) :
  unalias_rec_param S i k h body = Some (h', body') ->
  expr_equivalent_stated S (Let_cont_recursive [(k, h)] body)
    (Let_cont_recursive [(k, h')] body').
Proof.
  destruct h as [params metadata handler]. unfold unalias_rec_param.
  destruct (nth_error params i) as [p | ] eqn:Hparam; [ | discriminate ].
  destruct (nth_error (hm_params_kinds metadata) i) as [kind | ] eqn:Hkind;
    [ | discriminate ].
  destruct (first_given_var k i body) as [[x | ] | ]; try discriminate.
  destruct
    (negb (hm_is_exn_handler metadata) && all_give_var k i x true body
     && all_give_var k i p true handler
     && negb (among x (drop_nth i params))
     && negb (among p (drop_nth i params))
     && is_stated_at S (hm_params_subkinds metadata) i (Var x))
    eqn:Hcond; [ | discriminate ].
  intros [= <- <-].
  apply andb_true_iff in Hcond as [Hcond Hstated].
  apply andb_true_iff in Hcond as [Hcond Hfresh].
  apply andb_true_iff in Hcond as [Hcond Hx].
  apply andb_true_iff in Hcond as [Hcond Hturns].
  apply andb_true_iff in Hcond as [_ Hgive].
  apply negb_true_iff in Hfresh, Hx.
  rewrite all_give_var_simple in Hgive, Hturns.
  intros vars conts statics hp rd Hhold.
  exact
    (loop_param_equivalent i k (Var x) p kind params metadata handler body
       Hparam Hkind Hx Hfresh Hgive Hturns vars conts statics hp rd_
       (is_stated_at_given_is_stated statics S vars _ i (Var x) Hhold
          Hstated)).
Qed.

Lemma unalias_rec_from_equivalent (S : stated_in_scope) (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    expr_equivalent_stated S (Let_cont_recursive [(k, h)] body)
      (Let_cont_recursive [(k, fst (unalias_rec_from S i k h body))]
         (snd (unalias_rec_from S i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [unalias_rec_from].
  - apply expr_equivalent_stated_refl.
  - destruct (unalias_rec_param S i k h body) as [[h' body'] | ] eqn:Hparam;
      [ | apply IH ].
    exact (expr_equivalent_stated_trans S _ _ _
      (unalias_rec_param_equivalent S i k h body h' body' Hparam)
      (IH k h' body')).
Qed.

(* Each layer under what is known there, by the laws of
   FlambdaStatedContextProofs.v; then the parameters of a recursive
   [Let_cont] of one continuation are looked at. *)
Fixpoint unalias_rec_params_under_equivalent (e : expr) :
  ∀ S : stated_in_scope,
    expr_equivalent_stated S e (unalias_rec_params_under S e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros S; cbn [unalias_rec_params_under];
    try apply expr_equivalent_stated_refl.
  - apply let_equivalent_stated, unalias_rec_params_under_equivalent.
  - apply let_static_equivalent_stated, unalias_rec_params_under_equivalent.
  - apply (expr_equivalent_stated_trans S _
      (Let_cont_non_recursive k
         (Handler params metadata
            (unalias_rec_params_under
               (params_stated params (hm_params_subkinds metadata) S)
               handler))
         body)).
    + apply let_cont_handler_equivalent_stated,
        unalias_rec_params_under_equivalent.
    + apply let_cont_body_equivalent_stated,
        unalias_rec_params_under_equivalent.
  - assert (Hbody :
      expr_equivalent_stated S (Let_cont_recursive handlers body)
        (Let_cont_recursive handlers (unalias_rec_params_under S body))).
    { apply let_cont_recursive_body_equivalent_stated,
        unalias_rec_params_under_equivalent. }
    destruct handlers
      as [ | [k [params metadata handler]] [ | second handlers]];
      try exact Hbody.
    pose proof
      (unalias_rec_from_equivalent S (length params) k
         (Handler params metadata handler) (unalias_rec_params_under S body))
      as Hfrom.
    destruct
      (unalias_rec_from S (length params) k
         (Handler params metadata handler) (unalias_rec_params_under S body))
      as [h' body'].
    exact (expr_equivalent_stated_trans S _ _ _ Hbody Hfrom).
Qed.

Theorem unalias_rec_params_equivalent :
  unalias_rec_params_equivalent_statement.
Proof.
  intros e.
  exact
    (expr_equivalent_of_stated _ _
       (unalias_rec_params_under_equivalent e [])).
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The loop that a function with a self tail call has become
   (FlambdaLoopifySpec.looped_code): its parameters have the names of
   those of the function, and it is given, from outside, the variable
   of each name. The turn passes the first on, and it goes: the [Let]
   at the head of the handler binds it to the variable of its name
   from outside. The turn gives the second another variable, and it
   stays. *)
Example own_name_goes :
  unalias_rec_params
    (Let_cont_recursive
       [(1%positive,
         Handler [10%positive; 11%positive]
           (make_handler_metadata [Value; Value])
           (Apply_cont
              (Apply_cont_expr 1%positive [Var 10%positive; Var 12%positive]
                 None)))]
       (Apply_cont
          (Apply_cont_expr 1%positive [Var 10%positive; Var 11%positive]
             None)))
  = Let_cont_recursive
      [(1%positive,
        Handler [11%positive] (make_handler_metadata [Value])
          (Let 10%positive Value (Simple (Var 10%positive))
             (Apply_cont
                (Apply_cont_expr 1%positive [Var 12%positive] None))))]
      (Apply_cont (Apply_cont_expr 1%positive [Var 11%positive] None)).
Proof. reflexivity. Qed.
