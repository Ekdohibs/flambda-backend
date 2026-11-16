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

(* Proofs of the statements of FlambdaAlphaFreshSpec.v. Nothing here
   needs review.

   From left to right each statement goes through the comparison
   ([opened_alpha]): it accepts two alpha-equivalent terms, so it
   accepts the scopes of two binders under the pairs of their bound
   names; FlambdaCompareConverseProofs.opened_eqb then gives that it
   accepts the scopes, with no names paired, once the bound names of
   both are exchanged for the same fresh ones; and what it accepts is
   alpha-equivalent. From right to left each is the rule of
   FlambdaFreshBindersProofs.v, or for code that of
   FlambdaCompareConverseProofs.alpha_code_swap on both sides. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaBindersSpec FlambdaBindersProofs
  FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaSpec FlambdaAlphaProofs FlambdaCompareSpec
  FlambdaCompareProofs FlambdaCompareAlphaSpec FlambdaCompareAlphaProofs
  FlambdaFreshQuantifierSpec FlambdaFreshQuantifierProofs
  FlambdaFreshBindersSpec FlambdaFreshBindersProofs
  FlambdaCompareConverseSpec FlambdaCompareConverseProofs
  FlambdaAlphaFreshSpec.
Import ListNotations.

(* ======================================================================== *)
(* From the comparison to the scopes at fresh names                         *)
(* ======================================================================== *)

Lemma opened_alpha (xs xs' zs : list variable) (ks ks' zks : list continuation)
    (e e' : expr) (vm : name_map) :
  code_closed e = true ->
  code_closed e' = true ->
  binders_distinct e = true ->
  binders_distinct e' = true ->
  opens_at xs zs (free_variables_of_expr e) ->
  opens_at xs' zs (free_variables_of_expr e') ->
  opens_at ks zks (free_continuations_of_expr e) ->
  opens_at ks' zks (free_continuations_of_expr e') ->
  bind_names xs xs' [] = Some vm ->
  expr_eqb vm (combine ks ks') e e' = true ->
  alpha_equivalent (rename_expr (swap_binders xs zs ks zks) e)
    (rename_expr (swap_binders xs' zs ks' zks) e').
Proof.
  intros Hclosed Hclosed' Hdistinct Hdistinct' Hxs Hxs' Hks Hks' Hbind Heq.
  apply expr_eqb_alpha_equivalent.
  - rewrite code_closed_equivariant. exact Hclosed.
  - rewrite code_closed_equivariant. exact Hclosed'.
  - rewrite binders_distinct_equivariant. exact Hdistinct.
  - rewrite binders_distinct_equivariant. exact Hdistinct'.
  - exact
      (opened_eqb xs xs' zs ks ks' zks e e' vm Hclosed Hclosed' Hxs Hxs' Hks
         Hks' Hbind Heq).
Qed.

(* One bound name and one new name. *)
Lemma opens_at_single (bound fresh : positive) (free : list positive) :
  among fresh (bound :: free) = false -> opens_at [bound] [fresh] free.
Proof.
  intros Houtside.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ]. intros name Hname.
  destruct (Pos.eqb_spec name fresh) as [-> | Hdifferent];
    [ exact Houtside | ].
  apply Pos.eqb_neq in Hdifferent.
  rewrite among_cons, Hdifferent in Hname. discriminate Hname.
Qed.

(* ======================================================================== *)
(* A non-recursive [Let_cont]                                               *)
(* ======================================================================== *)

Lemma body_eqb_fresh (k k' : continuation) (body body' : expr) :
  code_closed body = true ->
  code_closed body' = true ->
  binders_distinct body = true ->
  binders_distinct body' = true ->
  expr_eqb [] [(k, k')] body body' = true ->
  И fresh,
    alpha_equivalent (rename_expr (swap_continuation k fresh) body)
      (rename_expr (swap_continuation k' fresh) body').
Proof.
  intros Hclosed Hclosed' Hdistinct Hdistinct' Heq.
  exists
    ((k :: free_continuations_of_expr body)
     ++ k' :: free_continuations_of_expr body').
  intros fresh Hfresh. apply among_app_false in Hfresh as [Hfresh Hfresh'].
  apply (opened_alpha [] [] [] [k] [k'] [fresh] body body' []);
    solve [ assumption | apply opens_at_nil | reflexivity
          | apply opens_at_single; assumption ].
Qed.

Lemma handler_eqb_alpha_handlers (h h' : continuation_handler) :
  handler_code_closed h = true ->
  handler_code_closed h' = true ->
  handler_binders_distinct h = true ->
  handler_binders_distinct h' = true ->
  handler_eqb [] [] h h' = true ->
  alpha_handlers h h'.
Proof.
  destruct h as [params metadata handler],
    h' as [params' metadata' handler'].
  cbn [handler_code_closed handler_binders_distinct handler_eqb
       alpha_handlers].
  intros Hclosed Hclosed' Hdistinct Hdistinct' Heq.
  destruct (bind_names params params' []) as [vm | ] eqn:Hbind;
    [ | discriminate Heq ].
  apply andb_true_iff in Heq as [Hmetadata Heq].
  apply handler_metadata_eqb_eq in Hmetadata.
  apply andb_true_iff in Hdistinct as [Hparams Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hparams' Hdistinct'].
  pose proof (bind_names_length _ _ _ _ Hbind) as Hlength.
  split; [ exact Hlength | ]. split; [ exact Hmetadata | ].
  exists
    ((params ++ free_variables_of_expr handler)
     ++ params' ++ free_variables_of_expr handler').
  intros zs Hcount Houtside.
  apply distinct_outside_app in Houtside as [Houtside Houtside'].
  apply (opened_alpha params params' zs [] [] [] handler handler' vm);
    try solve [ assumption | apply opens_at_nil ].
  - split; [ exact Hparams | ]. split; [ exact Hcount | exact Houtside ].
  - split; [ exact Hparams' | ].
    split; [ exact (eq_trans Hcount Hlength) | exact Houtside' ].
Qed.

Lemma alpha_handlers_bound (h h' : continuation_handler) :
  handler_code_closed h = true ->
  handler_code_closed h' = true ->
  handler_binders_distinct h = true ->
  handler_binders_distinct h' = true ->
  alpha_handlers h h' ->
  ∀ (k : continuation) (body : expr),
    alpha_equivalent (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k h' body).
Proof.
  destruct h as [params metadata handler],
    h' as [params' metadata' handler'].
  cbn [handler_code_closed handler_binders_distinct alpha_handlers].
  intros Hclosed Hclosed' Hdistinct Hdistinct' (Hlength & <- & Hfresh) k body.
  apply andb_true_iff in Hdistinct as [Hparams _].
  apply andb_true_iff in Hdistinct' as [Hparams' _].
  exact
    (alpha_handler_fresh k params params' metadata handler handler' body
       Hclosed Hclosed' Hparams Hparams' Hlength Hfresh).
Qed.

Theorem alpha_let_cont_body_iff : alpha_let_cont_body_iff_statement.
Proof.
  intros k k' h body body' Hclosed Hclosed' Hdistinct Hdistinct'. split.
  - intros Halpha. apply alpha_equivalent_accepted in Halpha.
    cbn [expr_eqb] in Halpha. apply andb_true_iff in Halpha as [_ Hbody].
    apply body_eqb_fresh; assumption.
  - exact (alpha_let_cont_fresh k k' h body body' Hclosed Hclosed').
Qed.

Theorem alpha_handler_iff : alpha_handler_iff_statement.
Proof.
  intros k h h' body Hclosed Hclosed' Hdistinct Hdistinct'. split.
  - intros Halpha. apply alpha_equivalent_accepted in Halpha.
    cbn [expr_eqb] in Halpha. apply andb_true_iff in Halpha as [Hh _].
    apply handler_eqb_alpha_handlers; assumption.
  - intros Hh. apply alpha_handlers_bound; assumption.
Qed.

(* The handlers [k (x, x) = jump x] and [k (y, y') = jump y'], the jump
   being to the same continuation. *)
Theorem alpha_handlers_repeated_parameter
  : alpha_handlers_repeated_parameter_statement.
Proof.
  set (metadata := make_handler_metadata [Value; Value]).
  exists
    (Handler [1; 1] metadata
       (Apply_cont (Apply_cont_expr 1 [Var 1] None)))%positive,
    (Handler [2; 3] metadata
       (Apply_cont (Apply_cont_expr 1 [Var 3] None)))%positive.
  split; [ | split; [ reflexivity | split; [ reflexivity | ] ] ].
  2: { intros k body Hdistinct Halpha.
       apply alpha_equivalent_binders_distinct in Halpha.
       cbn [binders_distinct handler_binders_distinct] in Halpha.
       rewrite Hdistinct in Halpha. discriminate Halpha. }
  cbn [alpha_handlers]. split; [ reflexivity | ]. split; [ reflexivity | ].
  exists [1; 2; 3]%positive. intros zs Hcount [Hdistinct Houtside].
  destruct zs as [ | a [ | b [ | other zs]]]; try discriminate Hcount.
  assert (Hb : among b [1; 2; 3]%positive = false).
  { apply Houtside. rewrite !among_cons, Pos.eqb_refl, orb_true_r.
    reflexivity. }
  cbn [distinct] in Hdistinct. rewrite among_cons in Hdistinct.
  apply andb_true_iff in Hdistinct as [Hab _].
  apply negb_true_iff in Hab. apply orb_false_iff in Hab as [Hab _].
  rewrite !among_cons in Hb.
  apply orb_false_iff in Hb as [Hb1 Hb]. apply orb_false_iff in Hb as [Hb2 _].
  unfold swap_variables. cbn [rename_expr].
  unfold rename_apply_cont, rename_simple, rename_variable,
    rename_continuation.
  cbn [map ac_args ac_continuation ac_trap_action option_map
       variable_permutation continuation_permutation combine permute].
  unfold swap. rewrite !Pos.eqb_refl.
  rewrite (Pos.eqb_sym a b) in Hab. rewrite Hb1, Hb2, Hab.
  apply Alpha_refl.
Qed.

Theorem alpha_let_cont_iff : alpha_let_cont_iff_statement.
Proof.
  intros k k' h h' body body' Hclosed Hclosed' Hdistinct Hdistinct'.
  cbn [code_closed binders_distinct] in Hclosed, Hclosed', Hdistinct,
    Hdistinct'.
  apply andb_true_iff in Hclosed as [Hclosed_h Hclosed].
  apply andb_true_iff in Hclosed' as [Hclosed_h' Hclosed'].
  apply andb_true_iff in Hdistinct as [Hdistinct_h Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hdistinct_h' Hdistinct'].
  split.
  - intros Halpha. apply alpha_equivalent_accepted in Halpha.
    cbn [expr_eqb] in Halpha. apply andb_true_iff in Halpha as [Hh Hbody].
    split;
      [ apply handler_eqb_alpha_handlers | apply body_eqb_fresh ];
      assumption.
  - intros [Hh Hbody].
    apply Alpha_trans with (Let_cont_non_recursive k h' body);
      [ apply alpha_handlers_bound; assumption | ].
    exact (alpha_let_cont_fresh k k' h' body body' Hclosed Hclosed' Hbody).
Qed.

(* ======================================================================== *)
(* Code                                                                     *)
(* ======================================================================== *)

Lemma code_eqb_alpha_code (c c' : code) :
  code_is_closed c = true ->
  code_is_closed c' = true ->
  code_binders_distinct c = true ->
  code_binders_distinct c' = true ->
  code_eqb c c' = true ->
  alpha_code c c'.
Proof.
  intros Hclosed Hclosed' Hdistinct Hdistinct' Heq.
  destruct (proj1 (closed_code c) Hclosed) as [_ Hclosed_body].
  destruct (proj1 (closed_code c') Hclosed') as [_ Hclosed_body'].
  destruct c as [params my_closure ret exn metadata body],
    c' as [params' my_closure' ret' exn' metadata' body'].
  cbn [code_body] in Hclosed_body, Hclosed_body'.
  cbn [code_eqb] in Heq.
  destruct (bind_names params params' [(my_closure, my_closure')])
    as [vm | ] eqn:Hbind; [ | discriminate Heq ].
  apply andb_true_iff in Heq as [Hmetadata Heq].
  apply code_metadata_eqb_eq in Hmetadata.
  cbn [code_binders_distinct] in Hdistinct, Hdistinct'.
  apply andb_true_iff in Hdistinct as [Hbinders Hdistinct].
  apply andb_true_iff in Hbinders as [Hvariables Hcontinuations].
  apply andb_true_iff in Hdistinct' as [Hbinders' Hdistinct'].
  apply andb_true_iff in Hbinders' as [Hvariables' Hcontinuations'].
  pose proof (bind_names_length _ _ _ _ Hbind) as Hlength.
  cbn [alpha_code]. split; [ exact Hlength | ].
  split; [ exact Hmetadata | ].
  exists
    (((my_closure :: params) ++ free_variables_of_expr body)
     ++ (my_closure' :: params') ++ free_variables_of_expr body').
  intros zs Hcount Houtside.
  exists
    (([ret; exn] ++ free_continuations_of_expr body)
     ++ [ret'; exn'] ++ free_continuations_of_expr body').
  intros zks Hcount_continuations Houtside_continuations.
  apply distinct_outside_app in Houtside as [Houtside Houtside'].
  apply distinct_outside_app in Houtside_continuations
    as [Houtside_continuations Houtside_continuations'].
  apply (opened_alpha (my_closure :: params) (my_closure' :: params') zs
           [ret; exn] [ret'; exn'] zks body body' vm);
    try assumption.
  - split; [ exact Hvariables | ]. split; [ exact Hcount | exact Houtside ].
  - split; [ exact Hvariables' | ].
    split; [ exact (eq_trans Hcount (f_equal S Hlength)) | exact Houtside' ].
  - split; [ exact (distinct_two _ _ Hcontinuations) | ].
    split; [ exact Hcount_continuations | exact Houtside_continuations ].
  - split; [ exact (distinct_two _ _ Hcontinuations') | ].
    split; [ exact Hcount_continuations | exact Houtside_continuations' ].
Qed.

Lemma alpha_code_bound (c c' : code) :
  code_is_closed c = true ->
  code_is_closed c' = true ->
  code_binders_distinct c = true ->
  code_binders_distinct c' = true ->
  alpha_code c c' ->
  ∀ (before : list static_binding) (id : code_id)
    (after : list static_binding) (body : expr),
    alpha_equivalent (Let_static (before ++ Bind_code id c :: after) body)
      (Let_static (before ++ Bind_code id c' :: after) body).
Proof.
  intros Hclosed Hclosed' Hdistinct Hdistinct' Halpha before id after body.
  destruct c as [params my_closure ret exn metadata function_body],
    c' as [params' my_closure' ret' exn' metadata' function_body'].
  cbn [alpha_code] in Halpha. destruct Halpha as (Hlength & <- & Hfresh).
  cbn [code_binders_distinct] in Hdistinct, Hdistinct'.
  apply andb_true_iff in Hdistinct as [Hbinders _].
  apply andb_true_iff in Hbinders as [Hvariables Hcontinuations].
  apply andb_true_iff in Hdistinct' as [Hbinders' _].
  apply andb_true_iff in Hbinders' as [Hvariables' Hcontinuations'].
  destruct
    (for_fresh_names_elim _ _
       (((my_closure :: params) ++ free_variables_of_expr function_body)
        ++ (my_closure' :: params') ++ free_variables_of_expr function_body')
       Hfresh)
    as (zs & Hcount & Houtside & Hfresh_continuations).
  destruct
    (for_fresh_names_elim _ _
       (([ret; exn] ++ free_continuations_of_expr function_body)
        ++ [ret'; exn'] ++ free_continuations_of_expr function_body')
       Hfresh_continuations)
    as (zks & Hcount_continuations & Houtside_continuations & Hscopes).
  apply distinct_outside_app in Houtside as [Houtside Houtside'].
  apply distinct_outside_app in Houtside_continuations
    as [Houtside_continuations Houtside_continuations'].
  destruct zs as [ | new_my_closure new_params]; [ discriminate Hcount | ].
  destruct zks as [ | new_ret [ | new_exn [ | other zks]]];
    try discriminate Hcount_continuations.
  assert (Hxs :
    opens_at (my_closure :: params) (new_my_closure :: new_params)
      (free_variables_of_expr function_body)).
  { split; [ exact Hvariables | ]. split; [ exact Hcount | exact Houtside ]. }
  assert (Hxs' :
    opens_at (my_closure' :: params') (new_my_closure :: new_params)
      (free_variables_of_expr function_body')).
  { split; [ exact Hvariables' | ].
    split; [ exact (eq_trans Hcount (f_equal S Hlength)) | exact Houtside' ]. }
  assert (Hks :
    opens_at [ret; exn] [new_ret; new_exn]
      (free_continuations_of_expr function_body)).
  { split; [ exact (distinct_two _ _ Hcontinuations) | ].
    split; [ reflexivity | exact Houtside_continuations ]. }
  assert (Hks' :
    opens_at [ret'; exn'] [new_ret; new_exn]
      (free_continuations_of_expr function_body')).
  { split; [ exact (distinct_two _ _ Hcontinuations') | ].
    split; [ reflexivity | exact Houtside_continuations' ]. }
  apply Alpha_trans
    with (Let_static
            (before
             ++ Bind_code id
                  (Code new_params new_my_closure new_ret new_exn metadata
                     (rename_expr
                        (opening (my_closure :: params)
                           (new_my_closure :: new_params) [ret; exn]
                           [new_ret; new_exn])
                        function_body))
             :: after)
            body);
    [ apply alpha_code_swap; assumption | ].
  apply Alpha_trans
    with (Let_static
            (before
             ++ Bind_code id
                  (Code new_params new_my_closure new_ret new_exn metadata
                     (rename_expr
                        (opening (my_closure' :: params')
                           (new_my_closure :: new_params) [ret'; exn']
                           [new_ret; new_exn])
                        function_body'))
             :: after)
            body);
    [ | apply Alpha_sym, alpha_code_swap; assumption ].
  exact
    (Alpha_context
       (Let_static_code before id new_params new_my_closure new_ret new_exn
          metadata Hole after body)
       _ _ Hscopes).
Qed.

Lemma forall2b_middle {A : Type} (f : A -> A -> bool) (before : list A)
    (a a' : A) (after after' : list A) :
  forall2b f (before ++ a :: after) (before ++ a' :: after') = true ->
  f a a' = true.
Proof.
  induction before as [ | first before IH]; cbn [app forall2b];
    intros Heq; apply andb_true_iff in Heq as [Hfirst Hrest];
    [ exact Hfirst | exact (IH Hrest) ].
Qed.

Theorem alpha_code_iff : alpha_code_iff_statement.
Proof.
  intros before id c c' after body Hclosed Hclosed' Hdistinct Hdistinct'.
  split.
  - intros Halpha. apply alpha_equivalent_accepted in Halpha.
    cbn [expr_eqb] in Halpha. apply andb_true_iff in Halpha as [Hgroup _].
    apply forall2b_middle in Hgroup. cbn [binding_eqb] in Hgroup.
    apply andb_true_iff in Hgroup as [_ Hcode].
    apply code_eqb_alpha_code; assumption.
  - intros Hcode. apply alpha_code_bound; assumption.
Qed.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

Lemma binding_eqb_alpha_binding (binding binding' : static_binding) :
  binding_code_closed binding = true ->
  binding_code_closed binding' = true ->
  binding_binders_distinct binding = true ->
  binding_binders_distinct binding' = true ->
  binding_eqb [] binding binding' = true ->
  alpha_binding binding binding'.
Proof.
  intros Hclosed Hclosed' Hdistinct Hdistinct' Heq.
  destruct binding as [id c | id | s const],
    binding' as [id' c' | id' | s' const'];
    cbn [binding_eqb alpha_binding binding_code_closed
         binding_binders_distinct] in *;
    try discriminate Heq.
  - apply andb_true_iff in Heq as [Hid Hcode]. apply Pos.eqb_eq in Hid.
    split; [ exact Hid | apply code_eqb_alpha_code; assumption ].
  - apply Pos.eqb_eq in Heq. exact Heq.
  - apply andb_true_iff in Heq as [Hs Hconst]. apply Pos.eqb_eq in Hs.
    split; [ exact Hs | exact (static_const_eqb_nil _ _ Hconst) ].
Qed.

Lemma bindings_eqb_alpha_bindings (bindings : list static_binding) :
  ∀ bindings' : list static_binding,
    forallb binding_code_closed bindings = true ->
    forallb binding_code_closed bindings' = true ->
    forallb binding_binders_distinct bindings = true ->
    forallb binding_binders_distinct bindings' = true ->
    forall2b (binding_eqb []) bindings bindings' = true ->
    Forall2 alpha_binding bindings bindings'.
Proof.
  induction bindings as [ | binding bindings IH];
    intros [ | binding' bindings'] Hclosed Hclosed' Hdistinct Hdistinct' Heq;
    cbn [forall2b] in Heq; try discriminate Heq; [ constructor | ].
  cbn [forallb] in Hclosed, Hclosed', Hdistinct, Hdistinct'.
  apply andb_true_iff in Hclosed as [Hclosed_first Hclosed].
  apply andb_true_iff in Hclosed' as [Hclosed_first' Hclosed'].
  apply andb_true_iff in Hdistinct as [Hdistinct_first Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hdistinct_first' Hdistinct'].
  apply andb_true_iff in Heq as [Hfirst Hrest].
  constructor;
    [ apply binding_eqb_alpha_binding; assumption | apply IH; assumption ].
Qed.

(* One binding after another: [done] is what has been replaced. *)
Lemma alpha_bindings_bound (body : expr)
    (bindings bindings' : list static_binding) :
  Forall2 alpha_binding bindings bindings' ->
  ∀ done : list static_binding,
    forallb binding_code_closed bindings = true ->
    forallb binding_code_closed bindings' = true ->
    forallb binding_binders_distinct bindings = true ->
    forallb binding_binders_distinct bindings' = true ->
    alpha_equivalent (Let_static (done ++ bindings) body)
      (Let_static (done ++ bindings') body).
Proof.
  intros Hall.
  induction Hall as [ | binding binding' bindings bindings' Hbinding _ IH];
    intros done Hclosed Hclosed' Hdistinct Hdistinct'; [ apply Alpha_refl | ].
  cbn [forallb] in Hclosed, Hclosed', Hdistinct, Hdistinct'.
  apply andb_true_iff in Hclosed as [Hclosed_first Hclosed].
  apply andb_true_iff in Hclosed' as [Hclosed_first' Hclosed'].
  apply andb_true_iff in Hdistinct as [Hdistinct_first Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hdistinct_first' Hdistinct'].
  apply Alpha_trans with (Let_static (done ++ binding' :: bindings) body).
  - destruct binding as [id c | id | s const],
      binding' as [id' c' | id' | s' const'];
      cbn [alpha_binding binding_code_closed binding_binders_distinct] in *;
      try contradiction.
    + destruct Hbinding as [<- Hcode]. apply alpha_code_bound; assumption.
    + subst id'. apply Alpha_refl.
    + destruct Hbinding as [<- <-]. apply Alpha_refl.
  - replace (done ++ binding' :: bindings)
      with ((done ++ [binding']) ++ bindings)
      by (rewrite <- app_assoc; reflexivity).
    replace (done ++ binding' :: bindings')
      with ((done ++ [binding']) ++ bindings')
      by (rewrite <- app_assoc; reflexivity).
    apply IH; assumption.
Qed.

Theorem alpha_let_static_iff : alpha_let_static_iff_statement.
Proof.
  intros bindings bindings' body body' Hclosed Hclosed' Hdistinct Hdistinct'.
  cbn [code_closed binders_distinct] in Hclosed, Hclosed', Hdistinct,
    Hdistinct'.
  apply andb_true_iff in Hclosed as [Hclosed_group Hclosed].
  apply andb_true_iff in Hclosed' as [Hclosed_group' Hclosed'].
  apply andb_true_iff in Hdistinct as [Hdistinct_group Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hdistinct_group' Hdistinct'].
  split.
  - intros Halpha. apply alpha_equivalent_accepted in Halpha.
    cbn [expr_eqb] in Halpha. apply andb_true_iff in Halpha as [Hgroup Hbody].
    split;
      [ apply bindings_eqb_alpha_bindings
      | apply expr_eqb_alpha_equivalent ];
      assumption.
  - intros [Hgroup Hbody].
    apply Alpha_trans with (Let_static bindings body').
    + exact (Alpha_context (Let_static_body bindings Hole) _ _ Hbody).
    + exact
        (alpha_bindings_bound body' bindings bindings' Hgroup []
           Hclosed_group Hclosed_group' Hdistinct_group Hdistinct_group').
Qed.
