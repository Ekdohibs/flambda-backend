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
  FlambdaUntagParamsNamesProofs FlambdaTwinParamsSpec FlambdaTwinParamsProofs.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite commutes with renaming                                       *)
(* ======================================================================== *)

Lemma alias_of_renamed (r : renaming) (d : named) :
  alias_of (rename_named r d) = option_map (rename_variable r) (alias_of d).
Proof. destruct d as [[x | sym | c] | | | | | | ]; reflexivity. Qed.

Lemma twin_simple_renamed (r : renaming) (pairs : twinned) (s : simple) :
  twin_simple (rename_tags r pairs) (rename_simple r s)
  = rename_simple r (twin_simple pairs s).
Proof.
  destruct s as [x | sym | c]; [ | reflexivity | reflexivity ].
  cbn [rename_simple twin_simple]. rewrite lookup_rename_tags.
  destruct (lookup x pairs); reflexivity.
Qed.

Lemma twin_args_renamed (r : renaming) (ts : list nat) (pairs : twinned)
    (args : list simple) :
  twin_args ts (rename_tags r pairs) (map (rename_simple r) args)
  = map (rename_simple r) (twin_args ts pairs args).
Proof.
  unfold twin_args. rewrite moved_from_map, map_app, !map_map. f_equal.
  apply map_ext. intros s. apply twin_simple_renamed.
Qed.

Lemma twin_apply_cont_renamed (r : renaming) (active : positions)
    (pairs : twinned) (ac : apply_cont) :
  twin_apply_cont (rename_keyed r active) (rename_tags r pairs)
    (rename_apply_cont r ac)
  = rename_apply_cont r (twin_apply_cont active pairs ac).
Proof.
  unfold twin_apply_cont, rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite positions_of_renamed, twin_args_renamed. reflexivity.
Qed.

Lemma twin_named_renamed (r : renaming) (pairs : twinned) (d : named) :
  twin_named (rename_tags r pairs) (rename_named r d)
  = rename_named r (twin_named pairs d).
Proof.
  unfold twin_named. rewrite untag_of_renamed.
  destruct (untag_of d) as [v | ]; cbn [option_map]; [ | reflexivity ].
  rewrite lookup_rename_tags. destruct (lookup v pairs); reflexivity.
Qed.

Lemma twin_names_renamed (r : renaming) (e : expr) :
  ∀ pending : list variable,
    twin_names (map (rename_variable r) pending) (rename_expr r e)
    = map (rename_variable r) (twin_names pending e).
Proof.
  induction e
    as [x kind d body IH | bindings body IH | k h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    intros [ | p rest]; cbn [map rename_expr twin_names]; try reflexivity.
  rewrite alias_of_renamed.
  destruct (alias_of d) as [q | ]; cbn [option_map]; [ | reflexivity ].
  rewrite rename_variable_eqb. destruct (Pos.eqb q p).
  - cbn [map]. rewrite IH. reflexivity.
  - exact (IH (p :: rest)).
Qed.

Lemma renamed_from_renamed (r : renaming) (ts : list nat)
    (params : list variable) :
  ∀ (i : nat) (names : list variable),
    renamed_from ts i (map (rename_variable r) params)
      (map (rename_variable r) names)
    = map (rename_variable r) (renamed_from ts i params names).
Proof.
  induction params as [ | p params IH]; intros i names; [ reflexivity | ].
  cbn [map renamed_from].
  destruct (at_position ts i); [ destruct names as [ | x names] | ];
    cbn [map]; f_equal.
  - exact (IH (S i) []).
  - exact (IH (S i) names).
  - exact (IH (S i) names).
Qed.

Lemma twin_params_list_renamed (r : renaming) (ts : list nat)
    (params : list variable) (handler : expr) :
  twin_params_list ts (map (rename_variable r) params) (rename_expr r handler)
  = map (rename_variable r) (twin_params_list ts params handler).
Proof.
  unfold twin_params_list.
  rewrite moved_from_map, twin_names_renamed, renamed_from_renamed, map_app.
  reflexivity.
Qed.

Fixpoint twin_expr_renamed (r : renaming) (sites active : positions)
    (pairs : twinned) (pending : list variable) (e : expr) {struct e} :
  twin_expr (rename_keyed r sites) (rename_keyed r active)
    (rename_tags r pairs) (map (rename_variable r) pending) (rename_expr r e)
  = rename_expr r (twin_expr sites active pairs pending e)
with twin_handler_renamed (r : renaming) (sites active : positions)
    (pairs : twinned) (ts : list nat) (h : continuation_handler) {struct h} :
  twin_handler (rename_keyed r sites) (rename_keyed r active)
    (rename_tags r pairs) ts (rename_handler r h)
  = rename_handler r (twin_handler sites active pairs ts h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr twin_expr]; try reflexivity.
    + pose proof (tags_without_renamed r [y] pairs) as Hwithout.
      cbn [map] in Hwithout.
      assert (Hplain :
        Let (rename_variable r y) kind
          (twin_named (rename_tags r pairs) (rename_named r d))
          (twin_expr (rename_keyed r sites) (rename_keyed r active)
             (tags_without [rename_variable r y] (rename_tags r pairs)) []
             (rename_expr r body))
        = rename_expr r
            (Let y kind (twin_named pairs d)
               (twin_expr sites active (tags_without [y] pairs) [] body))).
      { cbn [rename_expr]. rewrite twin_named_renamed, Hwithout. f_equal.
        exact (twin_expr_renamed r sites active (tags_without [y] pairs) []
                 body). }
      destruct pending as [ | p rest]; cbn [map]; [ exact Hplain | ].
      rewrite alias_of_renamed.
      destruct (alias_of d) as [q | ]; cbn [option_map]; [ | exact Hplain ].
      rewrite rename_variable_eqb. destruct (Pos.eqb q p).
      * rewrite Hwithout.
        exact
          (twin_expr_renamed r sites active
             ((y, p) :: tags_without [y] pairs) rest body).
      * cbn [rename_expr]. rewrite Hwithout. f_equal.
        exact
          (twin_expr_renamed r sites active (tags_without [y] pairs)
             (p :: rest) body).
    + f_equal. exact (twin_expr_renamed r sites active pairs [] body).
    + rewrite positions_of_renamed, twin_handler_renamed. f_equal.
      exact
        (twin_expr_renamed r sites ((k', positions_of sites k') :: active)
           pairs [] body).
    + fold (rename_handlers r handlers). rewrite active_under_renamed.
      f_equal;
        [ | exact
              (twin_expr_renamed r sites (active_under sites handlers active)
                 pairs [] body) ].
      generalize (active_under sites handlers active). intros active'.
      unfold rename_handlers.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map].
      rewrite positions_of_renamed, twin_handler_renamed, IH. reflexivity.
    + rewrite twin_apply_cont_renamed. reflexivity.
    + f_equal. induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map fst snd]. rewrite twin_apply_cont_renamed, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler twin_handler].
    rewrite twin_params_list_renamed, tags_without_renamed, moved_from_map,
      twin_expr_renamed.
    reflexivity.
Qed.

(* ======================================================================== *)
(* The test commutes with renaming                                          *)
(* ======================================================================== *)

Lemma head_ok_renamed (r : renaming) (avoid : list variable) (e : expr) :
  ∀ pending : list variable,
    head_ok (map (rename_variable r) avoid) (map (rename_variable r) pending)
      (rename_expr r e)
    = head_ok avoid pending e.
Proof.
  induction e
    as [x kind d body IH | bindings body IH | k h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    intros [ | p rest]; cbn [map rename_expr head_ok]; try reflexivity.
  rewrite alias_of_renamed.
  destruct (alias_of d) as [q | ]; cbn [option_map]; [ | reflexivity ].
  rewrite rename_variable_eqb. destruct (Pos.eqb q p); [ apply IH | ].
  pose proof (IH (p :: rest)) as IH'. cbn [map] in IH'.
  rewrite !among_rename_variable, IH'. reflexivity.
Qed.

Lemma handler_twins_fit_renamed (r : renaming) (ts : list nat)
    (params : list variable) (handler : expr) :
  handler_twins_fit ts (map (rename_variable r) params)
    (rename_expr r handler)
  = handler_twins_fit ts params handler.
Proof.
  unfold handler_twins_fit.
  rewrite twin_params_list_renamed, distinct_renamed, moved_from_map,
    twin_names_renamed, <- map_app, head_ok_renamed.
  reflexivity.
Qed.

Fixpoint twin_ok_renamed (r : renaming) (sites active : positions)
    (e : expr) {struct e} :
  twin_ok (rename_keyed r sites) (rename_keyed r active) (rename_expr r e)
  = twin_ok sites active e
with twin_handler_ok_renamed (r : renaming) (sites active : positions)
    (ts : list nat) (h : continuation_handler) {struct h} :
  twin_handler_ok (rename_keyed r sites) (rename_keyed r active) ts
    (rename_handler r h)
  = twin_handler_ok sites active ts h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr twin_ok]; try reflexivity.
    + apply twin_ok_renamed.
    + apply twin_ok_renamed.
    + rewrite positions_of_renamed, twin_handler_ok_renamed.
      rewrite
        <- (twin_ok_renamed r sites ((k', positions_of sites k') :: active)).
      reflexivity.
    + fold (rename_handlers r handlers).
      rewrite active_under_renamed, twin_ok_renamed. f_equal.
      generalize (active_under sites handlers active). intros active'.
      unfold rename_handlers.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite positions_of_renamed, twin_handler_ok_renamed, IH. reflexivity.
    + cbn [rename_apply ap_continuation ap_exn_continuation].
      rewrite !no_positions_renamed. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler twin_handler_ok].
    rewrite handler_twins_fit_renamed, twin_ok_renamed. reflexivity.
Qed.

Fixpoint twin_sites_renamed (r : renaming) (sites : positions) (e : expr)
    {struct e} :
  twin_sites (rename_keyed r sites) (rename_expr r e)
  = rename_keyed r (twin_sites sites e)
with handler_twin_sites_renamed (r : renaming) (sites : positions)
    (k : continuation) (h : continuation_handler) {struct h} :
  handler_twin_sites (rename_keyed r sites) (rename_continuation r k)
    (rename_handler r h)
  = rename_keyed r (handler_twin_sites sites k h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr twin_sites]; try reflexivity.
    + apply twin_sites_renamed.
    + apply twin_sites_renamed.
    + rewrite handler_twin_sites_renamed, twin_sites_renamed, rename_keyed_app.
      reflexivity.
    + rewrite twin_sites_renamed, rename_keyed_app. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map flat_map].
      rewrite handler_twin_sites_renamed, IH, rename_keyed_app. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_twin_sites].
    rewrite positions_of_renamed, moved_from_map, !length_map,
      twin_sites_renamed.
    reflexivity.
Qed.

Lemma twin_untag_at_renamed (r : renaming) (sites : positions) (e : expr) :
  twin_untag_at (rename_keyed r sites) (rename_expr r e)
  = rename_expr r (twin_untag_at sites e).
Proof.
  unfold twin_untag_at.
  rewrite (twin_ok_renamed r sites [] e : twin_ok _ [] _ = _),
    twin_sites_renamed,
    (twin_expr_renamed r sites [] [] [] e : twin_expr _ [] [] [] _ = _),
    (untag_ok_renamed r (twin_sites sites e) [] [] []
       (twin_expr sites [] [] [] e)
     : untag_ok _ [] [] [] _ = _).
  destruct
    (twin_ok sites [] e
     && untag_ok (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e));
    [ | reflexivity ].
  exact
    (untag_expr_renamed r (twin_sites sites e) [] [] []
       (twin_expr sites [] [] [] e)).
Qed.

(* ======================================================================== *)
(* The positions that the validation uses commute with renaming             *)
(* ======================================================================== *)

Lemma after_alias_renamed (r : renaming) (p : variable) (e : expr) :
  after_alias (rename_variable r p) (rename_expr r e)
  = option_map (rename_expr r) (after_alias p e).
Proof.
  induction e
    as [x kind d body IH | bindings body IH | k h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    cbn [rename_expr after_alias]; try reflexivity.
  rewrite alias_of_renamed.
  destruct (alias_of d) as [q | ]; cbn [option_map]; [ | reflexivity ].
  rewrite rename_variable_eqb. destruct (Pos.eqb q p); [ reflexivity | ].
  exact IH.
Qed.

Lemma head_positions_renamed (r : renaming) (params : list variable) :
  ∀ (kinds : list flambda_kind) (i : nat) (e : expr),
    head_positions i (map (rename_variable r) params) kinds (rename_expr r e)
    = head_positions i params kinds e.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] i e;
    cbn [map head_positions]; try reflexivity.
  destruct (kind_eqb kind Value); [ | apply IH ].
  rewrite after_alias_renamed.
  destruct (after_alias p e) as [rest | ]; cbn [option_map]; [ | apply IH ].
  rewrite IH. reflexivity.
Qed.

Fixpoint twin_candidates_renamed (r : renaming) (e : expr) {struct e} :
  twin_candidates (rename_expr r e) = rename_keyed r (twin_candidates e)
with handler_twin_candidates_renamed (r : renaming) (k : continuation)
    (h : continuation_handler) {struct h} :
  handler_twin_candidates (rename_continuation r k) (rename_handler r h)
  = rename_keyed r (handler_twin_candidates k h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr twin_candidates]; try reflexivity.
    + apply twin_candidates_renamed.
    + apply twin_candidates_renamed.
    + rewrite handler_twin_candidates_renamed, twin_candidates_renamed,
        rename_keyed_app.
      reflexivity.
    + rewrite twin_candidates_renamed, rename_keyed_app. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map flat_map].
      rewrite handler_twin_candidates_renamed, IH, rename_keyed_app.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_twin_candidates].
    rewrite head_positions_renamed, twin_candidates_renamed. reflexivity.
Qed.

Fixpoint twin_blame_renamed (r : renaming) (sites active : positions)
    (e : expr) {struct e} :
  twin_blame (rename_keyed r sites) (rename_keyed r active) (rename_expr r e)
  = rename_keyed r (twin_blame sites active e)
with twin_handler_blame_renamed (r : renaming) (sites active : positions)
    (k : continuation) (h : continuation_handler) {struct h} :
  twin_handler_blame (rename_keyed r sites) (rename_keyed r active)
    (rename_continuation r k) (rename_handler r h)
  = rename_keyed r (twin_handler_blame sites active k h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr twin_blame]; try reflexivity.
    + apply twin_blame_renamed.
    + apply twin_blame_renamed.
    + rewrite positions_of_renamed, twin_handler_blame_renamed,
        rename_keyed_app.
      rewrite
        <- (twin_blame_renamed r sites
              ((k', positions_of sites k') :: active)).
      reflexivity.
    + fold (rename_handlers r handlers).
      rewrite active_under_renamed, twin_blame_renamed, rename_keyed_app.
      f_equal.
      generalize (active_under sites handlers active). intros active'.
      unfold rename_handlers.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map flat_map].
      rewrite twin_handler_blame_renamed, IH, rename_keyed_app. reflexivity.
    + cbn [rename_apply ap_continuation ap_exn_continuation].
      rewrite !all_of_renamed, rename_keyed_app. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler twin_handler_blame].
    rewrite positions_of_renamed, handler_twins_fit_renamed,
      twin_blame_renamed, rename_keyed_app.
    f_equal.
    destruct (handler_twins_fit (positions_of sites k) params handler);
      [ reflexivity | ].
    unfold rename_keyed. rewrite map_map. reflexivity.
Qed.

Lemma twin_origin_renamed (r : renaming) (sites new : positions)
    (k : continuation) (u : nat) :
  twin_origin (rename_keyed r sites) (rename_keyed r new)
    (rename_continuation r k, u)
  = rename_keyed r (twin_origin sites new (k, u)).
Proof.
  unfold twin_origin. rewrite !positions_of_renamed.
  destruct (positions_of new k) as [ | n rest]; [ reflexivity | ].
  destruct (nth_error (positions_of sites k) (u - n)); reflexivity.
Qed.

Lemma twin_origins_renamed (r : renaming) (sites new : positions)
    (refused : list (continuation * nat)) :
  flat_map (twin_origin (rename_keyed r sites) (rename_keyed r new))
    (rename_keyed r refused)
  = rename_keyed r (flat_map (twin_origin sites new) refused).
Proof.
  induction refused as [ | [k u] refused IH]; [ reflexivity | ].
  change (rename_keyed r ((k, u) :: refused))
    with ((rename_continuation r k, u) :: rename_keyed r refused).
  cbn [flat_map]. rewrite twin_origin_renamed, IH, rename_keyed_app.
  reflexivity.
Qed.

Lemma twin_settle_renamed (r : renaming) (fuel : nat) :
  ∀ (sites : positions) (e : expr),
    twin_settle fuel (rename_keyed r sites) (rename_expr r e)
    = rename_keyed r (twin_settle fuel sites e).
Proof.
  induction fuel as [ | fuel IH]; intros sites e; [ reflexivity | ].
  cbn [twin_settle].
  rewrite (twin_blame_renamed r sites [] e : twin_blame _ [] _ = _).
  destruct (twin_blame sites [] e) as [ | refused0 refused] eqn:Hblame.
  - cbn [rename_keyed map].
    rewrite twin_sites_renamed,
      (twin_expr_renamed r sites [] [] [] e : twin_expr _ [] [] [] _ = _),
      (blame_renamed r (twin_sites sites e) [] [] []
         (twin_expr sites [] [] [] e)
       : blame _ [] [] [] _ = _).
    destruct (blame (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e))
      as [ | b0 bs] eqn:Hb; [ reflexivity | ].
    exact
      (eq_trans
         (f_equal (fun s => twin_settle fuel s (rename_expr r e))
            (eq_trans
               (f_equal (fun l => prune l (rename_keyed r sites))
                  (twin_origins_renamed r sites (twin_sites sites e)
                     (b0 :: bs)))
               (prune_renamed r _ sites)))
         (IH _ e)).
  - cbn [rename_keyed map].
    exact
      (eq_trans
         (f_equal (fun s => twin_settle fuel s (rename_expr r e))
            (prune_renamed r (refused0 :: refused) sites))
         (IH (prune (refused0 :: refused) sites) e)).
Qed.

Lemma twin_positions_renamed (r : renaming) (e : expr) :
  twin_positions (rename_expr r e) = rename_keyed r (twin_positions e).
Proof.
  unfold twin_positions.
  rewrite twin_candidates_renamed, count_positions_renamed.
  apply twin_settle_renamed.
Qed.

Lemma twin_params_renamed : rewrite_equivariant twin_params.
Proof.
  intros r e. unfold twin_params.
  rewrite twin_candidates_renamed, count_positions_renamed.
  destruct (count_positions (twin_candidates e)); [ reflexivity | ].
  rewrite twin_positions_renamed. apply twin_untag_at_renamed.
Qed.

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
