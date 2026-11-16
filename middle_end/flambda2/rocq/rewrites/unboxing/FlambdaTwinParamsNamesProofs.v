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


(* FlambdaTwinParamsSpec.twin_params commutes with renaming and keeps
   scope: the lemmas from which FlambdaValidatorEquivarianceProofs.v
   and FlambdaStaticsScopeProofs.v prove the two statements. There is
   no statement here, and nothing here needs review.

   The pass compares names by equality only and makes none. So each of
   its functions gives, on a renamed term, the renamed result: the
   rewrite and the test for renamed positions, and the positions that
   the validation uses, which are computed from the term; what it
   takes from FlambdaUntagParamsSpec.v does too
   (FlambdaUntagParamsNamesProofs.v). For the scope: the variable of
   a [Let] that goes is a parameter of the handler whose head the
   [Let] was at; a twin that an argument or an untagging mentions is
   the parameter that the [Let] read, which was free there; and the
   new parameters of a handler are those that it had and those
   variables. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidatorEquivarianceBaseProofs FlambdaScopedProofs
  FlambdaUntagParamsSpec FlambdaUntagParamsProofs
  FlambdaUntagParamsNamesProofs FlambdaTwinParamsSpec FlambdaTwinParamsProofs
  FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite commutes with renaming                                       *)
(* ======================================================================== *)

(* The laws are proved by the tactic of FlambdaEquivarianceTactic.v,
   from the definitions. It is given: how the two maps of the pass
   are renamed; laws of FlambdaUntagParamsNamesProofs.v and
   FlambdaUntagParamsProofs.v for functions that the pass takes from
   FlambdaUntagParamsSpec.v; [twin_args] kept folded, which
   computation would unfold before its law applies; and the laws of
   the functions of the pass, stated one by one so that none is
   derived again where it is used. *)

#[local] Instance twinned_renaming : Renaming_of twinned := rename_tags.
#[local] Instance positions_renaming : Renaming_of positions := rename_keyed.

#[local] Arguments twin_args : simpl never.

(* The laws of the database are applied in one pass over the goal,
   which is faster than [autorewrite], in the proofs that pass with
   it: all but the last three. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Hint Rewrite @moved_from_map positions_of_renamed
  tags_without_renamed distinct_renamed count_positions_renamed
  : equivariance.

(* The law of [active_under], with the handlers renamed as computation
   writes them. *)
Lemma active_under_at_handlers (r : renaming) (sites active : positions)
    (handlers : list (continuation * continuation_handler)) :
  active_under (rename_keyed r sites)
    (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
       handlers)
    (rename_keyed r active)
  = rename_keyed r (active_under sites handlers active).
Proof. exact (active_under_renamed r sites active handlers). Qed.
#[local] Hint Rewrite active_under_at_handlers : equivariance.

Lemma alias_of_renamed : equivariance_of alias_of.
Proof. equivariance. Qed.
#[local] Hint Rewrite alias_of_renamed : equivariance.

Lemma twin_args_renamed : equivariance_of twin_args.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_args_renamed : equivariance.

Lemma twin_apply_cont_renamed : equivariance_of twin_apply_cont.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_apply_cont_renamed : equivariance.

Lemma twin_names_renamed (r : renaming) (e : expr) :
  ∀ pending : list variable,
    twin_names (map (rename_variable r) pending) (rename_expr r e)
    = map (rename_variable r) (twin_names pending e).
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_names_renamed : equivariance.

Lemma renamed_from_renamed (r : renaming) (ts : list nat)
    (params : list variable) :
  ∀ (i : nat) (names : list variable),
    renamed_from ts i (map (rename_variable r) params)
      (map (rename_variable r) names)
    = map (rename_variable r) (renamed_from ts i params names).
Proof. equivariance. Qed.
#[local] Hint Rewrite renamed_from_renamed : equivariance.

Lemma twin_params_list_renamed : equivariance_of twin_params_list.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_params_list_renamed : equivariance.

Lemma twin_expr_renamed : equivariance_of twin_expr.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_expr_renamed : equivariance.

Lemma head_ok_renamed (r : renaming) (avoid : list variable) (e : expr) :
  ∀ pending : list variable,
    head_ok (map (rename_variable r) avoid) (map (rename_variable r) pending)
      (rename_expr r e)
    = head_ok avoid pending e.
Proof. equivariance. Qed.
#[local] Hint Rewrite head_ok_renamed : equivariance.

(* The test of a handler asks [head_ok] with two lists appended to
   avoid: the database renames each part, and the law above is at one
   list renamed. *)
Lemma head_ok_app_renamed (r : renaming) (avoid avoid' pending : list variable)
    (e : expr) :
  head_ok (map (rename_variable r) avoid ++ map (rename_variable r) avoid')
    (map (rename_variable r) pending) (rename_expr r e)
  = head_ok (avoid ++ avoid') pending e.
Proof. rewrite <- map_app. apply head_ok_renamed. Qed.
#[local] Hint Rewrite head_ok_app_renamed : equivariance.

Lemma handler_twins_fit_renamed : equivariance_of handler_twins_fit.
Proof. equivariance. Qed.
#[local] Hint Rewrite handler_twins_fit_renamed : equivariance.

Lemma twin_ok_renamed : equivariance_of twin_ok.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_ok_renamed : equivariance.

Lemma twin_sites_renamed : equivariance_of twin_sites.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_sites_renamed : equivariance.

(* The pass starts its functions, and those of the other pass, at
   empty states. A law is used where the goal has its left-hand side
   as it is written, and the renaming of [[]] is written [[]]: so
   the laws at the empty states are stated. *)
Lemma twin_ok_at_start (r : renaming) (sites : positions) (e : expr) :
  twin_ok (rename_keyed r sites) [] (rename_expr r e) = twin_ok sites [] e.
Proof. exact (twin_ok_renamed r sites [] e). Qed.

Lemma twin_expr_at_start (r : renaming) (sites : positions) (e : expr) :
  twin_expr (rename_keyed r sites) [] [] [] (rename_expr r e)
  = rename_expr r (twin_expr sites [] [] [] e).
Proof. exact (twin_expr_renamed r sites [] [] [] e). Qed.

Lemma untag_ok_at_start (r : renaming) (sites : positions) (e : expr) :
  untag_ok (rename_keyed r sites) [] [] [] (rename_expr r e)
  = untag_ok sites [] [] [] e.
Proof. exact (untag_ok_renamed r sites [] [] [] e). Qed.

Lemma untag_expr_at_start (r : renaming) (sites : positions) (e : expr) :
  untag_expr (rename_keyed r sites) [] [] [] (rename_expr r e)
  = rename_expr r (untag_expr sites [] [] [] e).
Proof. exact (untag_expr_renamed r sites [] [] [] e). Qed.
#[local] Hint Rewrite twin_ok_at_start twin_expr_at_start untag_ok_at_start
  untag_expr_at_start : equivariance.

Lemma twin_untag_at_renamed : equivariance_of twin_untag_at.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_untag_at_renamed : equivariance.

(* ======================================================================== *)
(* The positions that the validation uses commute with renaming             *)
(* ======================================================================== *)

Lemma head_positions_renamed (r : renaming) (params : list variable) :
  ∀ (kinds : list flambda_kind) (i : nat) (e : expr),
    head_positions i (map (rename_variable r) params) kinds (rename_expr r e)
    = head_positions i params kinds e.
Proof. equivariance. Qed.
#[local] Hint Rewrite head_positions_renamed : equivariance.

Lemma twin_candidates_renamed : equivariance_of twin_candidates.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_candidates_renamed : equivariance.

Lemma twin_blame_renamed : equivariance_of twin_blame.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_blame_renamed : equivariance.

Lemma twin_blame_at_start (r : renaming) (sites : positions) (e : expr) :
  twin_blame (rename_keyed r sites) [] (rename_expr r e)
  = rename_keyed r (twin_blame sites [] e).
Proof. exact (twin_blame_renamed r sites [] e). Qed.

Lemma blame_at_start (r : renaming) (sites : positions) (e : expr) :
  blame (rename_keyed r sites) [] [] [] (rename_expr r e)
  = rename_keyed r (blame sites [] [] [] e).
Proof. exact (blame_renamed r sites [] [] [] e). Qed.
#[local] Hint Rewrite twin_blame_at_start blame_at_start : equivariance.

(* [twin_settle] goes on at the positions pruned by a list of refused
   positions that it has taken apart, or by the origins of one. The
   step for it: the list on the left is the list on the right renamed
   (the tactic shows that), and then the law of [prune] applies.
   [twin_settle] and [twin_params] are not proved with the one pass
   (FlambdaEquivarianceTactic.v says what it does not find): from
   here the database is applied by [autorewrite], as it is by
   default. *)
#[local] Ltac eqv_database ::= eqv_database_by_autorewrite.
#[local] Arguments prune : simpl never.
#[local] Arguments twin_origin : simpl never.
#[local] Ltac eqv_extra ::=
  lazymatch goal with
  | |- twin_settle ?fuel (prune ?l' (rename_keyed ?r ?sites)) _
       = _ (twin_settle ?fuel (prune ?l ?sites) _) =>
    replace l' with (rename_keyed r l);
    [ rewrite prune_renamed | symmetry ]
  end.

Lemma twin_settle_renamed : equivariance_of twin_settle.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_settle_renamed : equivariance.

Lemma twin_positions_renamed : equivariance_of twin_positions.
Proof. equivariance. Qed.
#[local] Hint Rewrite twin_positions_renamed : equivariance.

Lemma twin_params_renamed : rewrite_equivariant twin_params.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Lemma twin_names_length (e : expr) :
  ∀ pending : list variable,
    length (twin_names pending e) <= length pending.
Proof.
  induction e
    as [x kind d body IH | bindings body IH | k h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    intros [ | p rest]; cbn [twin_names length]; try lia.
  destruct (alias_of d) as [q | ]; [ | cbn [length]; lia ].
  destruct (Pos.eqb q p).
  - cbn [length]. specialize (IH rest). lia.
  - exact (IH (p :: rest)).
Qed.

(* The names are among the parameters renamed, if there are no more
   of them than of parameters moved. *)
Lemma in_names_renamed (ts : list nat) (x : variable)
    (params : list variable) :
  ∀ (i : nat) (names : list variable),
    length names <= length (moved_from ts i params) ->
    In x names ->
    In x (renamed_from ts i params names).
Proof.
  induction params as [ | p params IH]; intros i names;
    cbn [moved_from renamed_from].
  - destruct names; cbn [length]; [ intros _ H; exact H | lia ].
  - destruct (at_position ts i).
    + destruct names as [ | y names]; [ intros _ [] | ].
      cbn [length]. intros Hlength [<- | Hin]; [ left; reflexivity | ].
      right. apply (IH (S i) names); [ lia | exact Hin ].
    + intros Hlength Hin. right. exact (IH (S i) names Hlength Hin).
Qed.

Lemma twin_named_free_variables (pairs : twinned) (d : named)
    (a : variable) :
  In a (free_variables_of_named (twin_named pairs d)) ->
  In a (free_variables_of_named d) ∨ In a (map snd pairs).
Proof.
  unfold twin_named.
  destruct (untag_of d) as [v | ]; [ | intros H; left; exact H ].
  destruct (lookup v pairs) as [p | ] eqn:Hp; [ | intros H; left; exact H ].
  cbn. intros [<- | []]. right.
  exact (in_map snd pairs (v, p) (lookup_in v p pairs Hp)).
Qed.

(* A free variable of an application rewritten is one of the
   application, or a twin. *)
Lemma twin_apply_cont_free_variables (active : positions) (pairs : twinned)
    (ac : apply_cont) (a : variable) :
  In a (free_variables_of_apply_cont (twin_apply_cont active pairs ac)) ->
  In a (free_variables_of_apply_cont ac) ∨ In a (map snd pairs).
Proof.
  unfold free_variables_of_apply_cont, free_variables_of_simples,
    twin_apply_cont, twin_args.
  cbn [ac_args]. intros Ha. apply in_flat_map in Ha as (s & Hs & Ha).
  apply in_app_or in Hs as [Hs | Hs].
  - left. apply in_flat_map. exists s. split; assumption.
  - apply in_map_iff in Hs as (s0 & <- & Hs0).
    destruct s0 as [x | sym | c]; cbn [twin_simple] in Ha.
    + destruct (lookup x pairs) as [p | ] eqn:Hp.
      * right. cbn [free_variables_of_simple] in Ha.
        destruct Ha as [<- | []].
        exact (in_map snd pairs (x, p) (lookup_in x p pairs Hp)).
      * left. apply in_flat_map. eexists.
        split; [ exact (in_moved_from _ _ _ 0 Hs0) | exact Ha ].
    + left. apply in_flat_map. eexists.
      split; [ exact (in_moved_from _ _ _ 0 Hs0) | exact Ha ].
    + left. apply in_flat_map. eexists.
      split; [ exact (in_moved_from _ _ _ 0 Hs0) | exact Ha ].
Qed.

Fixpoint twin_expr_code_closed (sites active : positions) (pairs : twinned)
    (pending : list variable) (e : expr) {struct e} :
  code_closed (twin_expr sites active pairs pending e) = code_closed e
with twin_handler_code_closed (sites active : positions) (pairs : twinned)
    (ts : list nat) (h : continuation_handler) {struct h} :
  handler_code_closed (twin_handler sites active pairs ts h)
  = handler_code_closed h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [twin_expr code_closed]; try reflexivity.
    + destruct pending as [ | p rest];
        [ cbn [code_closed]; apply twin_expr_code_closed | ].
      destruct (alias_of d) as [q | ];
        [ | cbn [code_closed]; apply twin_expr_code_closed ].
      destruct (Pos.eqb q p); [ | cbn [code_closed] ];
        apply twin_expr_code_closed.
    + rewrite twin_expr_code_closed. reflexivity.
    + rewrite twin_handler_code_closed, twin_expr_code_closed. reflexivity.
    + rewrite twin_expr_code_closed. f_equal.
      generalize (active_under sites handlers active). intros active'.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb]. rewrite twin_handler_code_closed, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [twin_handler handler_code_closed]. apply twin_expr_code_closed.
Qed.

Fixpoint twin_expr_free_continuations (sites active : positions)
    (pairs : twinned) (pending : list variable) (e : expr) {struct e} :
  free_continuations_of_expr (twin_expr sites active pairs pending e)
  = free_continuations_of_expr e
with twin_handler_free_continuations (sites active : positions)
    (pairs : twinned) (ts : list nat) (h : continuation_handler) {struct h} :
  free_continuations_of_handler (twin_handler sites active pairs ts h)
  = free_continuations_of_handler h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [twin_expr free_continuations_of_expr]; try reflexivity.
    + destruct pending as [ | p rest];
        [ cbn [free_continuations_of_expr];
          apply twin_expr_free_continuations | ].
      destruct (alias_of d) as [q | ];
        [ | cbn [free_continuations_of_expr];
            apply twin_expr_free_continuations ].
      destruct (Pos.eqb q p); [ | cbn [free_continuations_of_expr] ];
        apply twin_expr_free_continuations.
    + rewrite twin_expr_free_continuations. reflexivity.
    + rewrite twin_handler_free_continuations, twin_expr_free_continuations.
      reflexivity.
    + rewrite twin_expr_free_continuations.
      generalize (active_under sites handlers active). intros active'.
      f_equal.
      * induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map fst]. rewrite IH. reflexivity.
      * f_equal.
        induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map flat_map].
        rewrite twin_handler_free_continuations, IH. reflexivity.
    + induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map flat_map snd]. rewrite IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [twin_handler free_continuations_of_handler].
    apply twin_expr_free_continuations.
Qed.

(* A free variable of the result is one of the term, or a twin, or
   the variable of a [Let] at the head that goes: the twins are the
   parameters that those [Let]s read, and their variables are
   parameters of the handler in the result. *)
Fixpoint twin_expr_free_variables (sites : positions) (a : variable)
    (e : expr) {struct e} :
  ∀ (active : positions) (pairs : twinned) (pending : list variable),
    In a (free_variables_of_expr (twin_expr sites active pairs pending e)) ->
    In a (free_variables_of_expr e)
    ∨ In a (map snd pairs)
    ∨ In a (twin_names pending e)
with twin_handler_free_variables (sites : positions) (a : variable)
    (h : continuation_handler) {struct h} :
  ∀ (active : positions) (pairs : twinned) (ts : list nat),
    In a (free_variables_of_handler (twin_handler sites active pairs ts h)) ->
    In a (free_variables_of_handler h) ∨ In a (map snd pairs).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros active pairs pending;
      cbn [twin_expr free_variables_of_expr];
      try (intros Hin; left; exact Hin).
    + (* a [Let] *)
      assert (Hunder :
        ∀ pending' : list variable,
          In a
            (without [y]
               (free_variables_of_expr
                  (twin_expr sites active (tags_without [y] pairs) pending'
                     body))) ->
          In a (free_variables_of_named d ++ without [y]
                  (free_variables_of_expr body))
          ∨ In a (map snd pairs)
          ∨ In a (twin_names pending' body)).
      { intros pending' Hin. apply in_without in Hin as [Hin Hy].
        destruct (twin_expr_free_variables sites a body _ _ _ Hin)
          as [Hin' | [Hin' | Hin']].
        - left. apply in_or_app. right. apply in_without. split; assumption.
        - right. left. exact (in_snd_tags_without [y] pairs a Hin').
        - right. right. exact Hin'. }
      assert (Hplain :
        In a
          (free_variables_of_expr
             (Let y kind (twin_named pairs d)
                (twin_expr sites active (tags_without [y] pairs) [] body))) ->
        In a (free_variables_of_named d ++ without [y]
                (free_variables_of_expr body))
        ∨ In a (map snd pairs)).
      { cbn [free_variables_of_expr]. intros Hin.
        apply in_app_or in Hin as [Hin | Hin].
        - destruct (twin_named_free_variables pairs d a Hin) as [Hd | Hp];
            [ left; apply in_or_app; left; exact Hd | right; exact Hp ].
        - destruct (Hunder [] Hin) as [Hin' | [Hin' | Hin']];
            [ left; exact Hin' | right; exact Hin' | ].
          rewrite twin_names_nil in Hin'. contradiction. }
      destruct pending as [ | p rest].
      { intros Hin. destruct (Hplain Hin) as [H | H];
          [ left; exact H | right; left; exact H ]. }
      cbn [twin_names].
      destruct (alias_of d) as [q | ] eqn:Halias.
      2: { intros Hin. destruct (Hplain Hin) as [H | H];
             [ left; exact H | right; left; exact H ]. }
      apply alias_of_some in Halias. subst d.
      destruct (Pos.eqb_spec q p) as [-> | Hne].
      * intros Hin.
        destruct (twin_expr_free_variables sites a body _ _ _ Hin)
          as [Hin' | [Hin' | Hin']].
        -- destruct (Pos.eq_dec a y) as [-> | Hay].
           ++ right. right. left. reflexivity.
           ++ left. apply in_or_app. right. apply in_without.
              split; [ exact Hin' | ].
              intros [Heq | []]. apply Hay. symmetry. exact Heq.
        -- cbn [map snd] in Hin'. destruct Hin' as [<- | Hin'].
           ++ left. apply in_or_app. left. left. reflexivity.
           ++ right. left. exact (in_snd_tags_without [y] pairs a Hin').
        -- right. right. right. exact Hin'.
      * cbn [free_variables_of_expr]. intros Hin.
        apply in_app_or in Hin as [Hin | Hin].
        -- left. apply in_or_app. left. exact Hin.
        -- exact (Hunder (p :: rest) Hin).
    + (* [Let_static] *)
      intros Hin. apply in_app_or in Hin as [Hin | Hin];
        [ left; apply in_or_app; left; exact Hin | ].
      destruct (twin_expr_free_variables sites a body _ _ _ Hin)
        as [Hin' | [Hin' | Hin']].
      * left. apply in_or_app. right. exact Hin'.
      * right. left. exact Hin'.
      * rewrite twin_names_nil in Hin'. contradiction.
    + (* [Let_cont_non_recursive] *)
      intros Hin. apply in_app_or in Hin as [Hin | Hin].
      * destruct (twin_handler_free_variables sites a hh _ _ _ Hin)
          as [Hin' | Hx]; [ | right; left; exact Hx ].
        left. apply in_or_app. left. exact Hin'.
      * destruct (twin_expr_free_variables sites a body _ _ _ Hin)
          as [Hin' | [Hin' | Hin']].
        -- left. apply in_or_app. right. exact Hin'.
        -- right. left. exact Hin'.
        -- rewrite twin_names_nil in Hin'. contradiction.
    + (* [Let_cont_recursive] *)
      intros Hin. apply in_app_or in Hin as [Hin | Hin].
      * assert (Hhandlers :
          In a (flat_map (fun '(_, h) => free_variables_of_handler h) handlers)
          ∨ In a (map snd pairs)).
        { revert Hin.
          generalize (active_under sites handlers active). intros active'.
          induction handlers as [ | [k' hh] handlers IH];
            cbn [map flat_map]; [ intros [] | ].
          intros Hin. apply in_app_or in Hin as [Hin | Hin].
          - destruct (twin_handler_free_variables sites a hh _ _ _ Hin)
              as [Hin' | Hx]; [ | right; exact Hx ].
            left. apply in_or_app. left. exact Hin'.
          - destruct (IH Hin) as [Hin' | Hx]; [ | right; exact Hx ].
            left. apply in_or_app. right. exact Hin'. }
        destruct Hhandlers as [Hin' | Hx]; [ | right; left; exact Hx ].
        left. apply in_or_app. left. exact Hin'.
      * destruct (twin_expr_free_variables sites a body _ _ _ Hin)
          as [Hin' | [Hin' | Hin']].
        -- left. apply in_or_app. right. exact Hin'.
        -- right. left. exact Hin'.
        -- rewrite twin_names_nil in Hin'. contradiction.
    + (* [Apply_cont] *)
      intros Hin.
      destruct (twin_apply_cont_free_variables active pairs ac a Hin)
        as [H | H]; [ left; exact H | right; left; exact H ].
    + (* [Switch] *)
      intros Hin. apply in_app_or in Hin as [Hin | Hin];
        [ left; apply in_or_app; left; exact Hin | ].
      assert (Harms :
        In a (flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms)
        ∨ In a (map snd pairs)).
      { induction arms as [ | [n ac] arms IH];
          cbn [map flat_map snd] in Hin |- *; [ contradiction | ].
        apply in_app_or in Hin as [Hin | Hin].
        - destruct (twin_apply_cont_free_variables active pairs ac a Hin)
            as [Hin' | Hx]; [ | right; exact Hx ].
          left. apply in_or_app. left. exact Hin'.
        - destruct (IH Hin) as [Hin' | Hx]; [ | right; exact Hx ].
          left. apply in_or_app. right. exact Hin'. }
      destruct Harms as [Hin' | Hx]; [ | right; left; exact Hx ].
      left. apply in_or_app. right. exact Hin'.
  - destruct h as [params metadata handler].
    intros active pairs ts.
    cbn [twin_handler free_variables_of_handler]. intros Hin.
    apply in_without in Hin as [Hin Hparams].
    destruct (twin_expr_free_variables sites a handler _ _ _ Hin)
      as [Hin' | [Hin' | Hin']].
    + left. apply in_without. split; [ exact Hin' | ].
      intros Hp. apply Hparams. apply among_In.
      apply among_twin_params. apply among_In. exact Hp.
    + right. exact (in_snd_tags_without params pairs a Hin').
    + exfalso. apply Hparams. unfold twin_params_list.
      apply in_or_app. left.
      apply in_names_renamed; [ apply twin_names_length | exact Hin' ].
Qed.

Lemma twin_expr_scoped (sites : positions) (e : expr) :
  scoped e (twin_expr sites [] [] [] e).
Proof.
  intros Hclosed. rewrite twin_expr_code_closed.
  split; [ exact Hclosed | split ].
  - intros a Ha.
    destruct (twin_expr_free_variables sites a e [] [] [] Ha)
      as [Hin | [[] | Hin]]; [ exact Hin | ].
    rewrite twin_names_nil in Hin. contradiction.
  - rewrite twin_expr_free_continuations. apply incl_refl.
Qed.

Lemma twin_untag_at_scoped (sites : positions) (e : expr) :
  scoped e (twin_untag_at sites e).
Proof.
  unfold twin_untag_at.
  destruct (twin_ok sites [] e); cbn [andb]; [ | apply scoped_refl ].
  destruct
    (untag_ok (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e))
    eqn:Hok;
    [ | apply scoped_refl ].
  apply (scoped_trans e (twin_expr sites [] [] [] e));
    [ apply twin_expr_scoped | ].
  pose proof (untag_at_scoped (twin_sites sites e) (twin_expr sites [] [] [] e))
    as H.
  unfold untag_at in H. rewrite Hok in H. exact H.
Qed.

Lemma twin_params_scoped (e : expr) : scoped e (twin_params e).
Proof.
  unfold twin_params.
  destruct (count_positions (twin_candidates e)); [ apply scoped_refl | ].
  apply twin_untag_at_scoped.
Qed.

Lemma twin_params_scope : keeps_scope twin_params.
Proof. apply keeps_scope_intro, twin_params_scoped. Qed.
