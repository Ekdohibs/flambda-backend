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


(* FlambdaUnboxTogetherSpec.unbox_together_knowing commutes with
   renaming and keeps scope: the lemmas from which
   FlambdaUnboxVariantsRenameProofs.v, FlambdaUnboxInScopeRenameProofs.v
   and FlambdaValidatorEquivarianceProofs.v prove the statements. There
   is no statement here, and nothing here needs review.

   The pass compares names by equality only and makes none. So each of
   its functions gives, on a renamed term and with what is known
   renamed, the renamed result: the rewrite and the test for renamed
   positions, and the positions that the pass uses, which are computed
   from the term; the sorts of box that a term unboxes are those of
   the term renamed. For the scope: the parameters of a handler are
   the same in another order; the [Let] of a parameter mentions the
   variable that the unboxing mentioned; and an argument is one that
   the application had, a constant, or a variable of what is known,
   which is in scope where the application is. *)

From Stdlib Require Import ZArith List Bool Arith Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaBlockLoadSpec FlambdaUnboxSpec
  FlambdaValidatorEquivarianceSpec FlambdaValidatorEquivarianceBaseProofs
  FlambdaScopedProofs FlambdaKeepsScopeProofs FlambdaUnboxParamsSpec
  FlambdaUnboxParamsRenameSpec FlambdaUnboxVariantsSpec
  FlambdaUnboxParamsNamesProofs FlambdaUntagParamsSpec
  FlambdaUntagParamsProofs FlambdaUntagParamsNamesProofs
  FlambdaUnboxTogetherSpec.
From OxCamlFlambda2 Require FlambdaUnboxTogetherProofs.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

Section Renamed.

(* The equations of FlambdaValidatorEquivarianceProofs.v that the
   proofs use, each under its name there, as in
   FlambdaUnboxParamsNamesProofs.v. *)
Hypothesis var_free_b_equivariant : var_free_b_equivariant_statement.

Hypothesis remove_unused_conts_equivariant :
  remove_unused_conts_equivariant_statement.

Hypothesis forget_vars_equivariant :
  ∀ (r : renaming) (xs : list variable) (known : known_blocks),
    forget_vars (map (rename_variable r) xs) (rename_known_blocks r known)
    = rename_known_blocks r (forget_vars xs known).

Hypothesis known_under_let_equivariant :
  ∀ (r : renaming) (known : known_blocks) (x : variable) (d : named),
    known_under_let (rename_known_blocks r known) (rename_variable r x)
      (rename_named r d)
    = rename_known_blocks r (known_under_let known x d).

Hypothesis known_under_statics_equivariant :
  ∀ (r : renaming) (known : known_blocks) (bindings : list static_binding),
    known_under_statics (rename_known_blocks r known)
      (map (rename_static_binding r) bindings)
    = rename_known_blocks r (known_under_statics known bindings).

Hypothesis known_block_of_equivariant :
  ∀ (r : renaming) (known : known_blocks) (s : simple),
    known_block_of (rename_known_blocks r known) (rename_simple r s)
    = option_map (rename_known_block r) (known_block_of known s).

Hypothesis field_loaded_equivariant :
  ∀ (r : renaming) (kind : block_access_kind) (field : Z)
    (block : known_block),
    field_loaded kind field (rename_known_block r block)
    = option_map (rename_simple r) (field_loaded kind field block).

Hypothesis load_known_named_equivariant :
  ∀ (r : renaming) (known : known_blocks) (d : named),
    load_known_named (rename_known_blocks r known) (rename_named r d)
    = rename_named r (load_known_named known d).

Hypothesis forget_wrapped_vars_equivariant :
  ∀ (r : renaming) (xs : list variable) (known : known_numbers),
    forget_wrapped_vars (map (rename_variable r) xs)
      (rename_known_numbers r known)
    = rename_known_numbers r (forget_wrapped_vars xs known).

Hypothesis numbers_under_let_equivariant :
  ∀ (r : renaming) (known : known_numbers) (x : variable) (d : named),
    numbers_under_let (rename_known_numbers r known) (rename_variable r x)
      (rename_named r d)
    = rename_known_numbers r (numbers_under_let known x d).

Hypothesis numbers_under_statics_equivariant :
  ∀ (r : renaming) (known : known_numbers) (bindings : list static_binding),
    numbers_under_statics (rename_known_numbers r known)
      (map (rename_static_binding r) bindings)
    = rename_known_numbers r (numbers_under_statics known bindings).

Hypothesis unwrapped_equivariant :
  ∀ (r : renaming) (known : known_numbers) (how : wrapping)
    (wrapped : simple),
    unwrapped (rename_known_numbers r known) how (rename_simple r wrapped)
    = option_map (rename_simple r) (unwrapped known how wrapped).

Hypothesis unbox_known_named_equivariant :
  ∀ (r : renaming) (known : known_numbers) (d : named),
    unbox_known_named (rename_known_numbers r known) (rename_named r d)
    = rename_named r (unbox_known_named known d).

(* The rewrite. *)

Lemma shown_number_renamed (r : renaming) (boxed : boxable_number)
    (unboxed : list variable) (known : facts) (s : simple) :
  shown_number boxed (map (rename_variable r) unboxed) (rename_facts r known)
    (rename_simple r s)
  = option_map (rename_simple r) (shown_number boxed unboxed known s).
Proof.
  unfold shown_number. rewrite certain_projection_equivariant by assumption.
  destruct
    (certain_projection known (Unbox_number boxed) (unboxed_kind boxed) s)
    as [number | ]; cbn [option_map]; [ | reflexivity ].
  rewrite
    (none_in_renamed r unboxed (fun p => var_in_simple p number)
       (fun p => var_in_simple p (rename_simple r number))
       (fun p => var_in_simple_equivariant r p number)).
  destruct (none_in unboxed (fun p => var_in_simple p number)); reflexivity.
Qed.

Lemma unboxed_simple_renamed (r : renaming) (boxed : boxable_number)
    (unboxed : list variable) (known : facts) (s : simple) :
  unboxed_simple boxed (map (rename_variable r) unboxed)
    (rename_facts r known) (rename_simple r s)
  = option_map (rename_simple r) (unboxed_simple boxed unboxed known s).
Proof.
  destruct s as [x | sym | c]; cbn [rename_simple unboxed_simple].
  - rewrite among_rename_variable.
    destruct (among x unboxed); [ reflexivity | ].
    exact (shown_number_renamed r boxed unboxed known (Var x)).
  - exact (shown_number_renamed r boxed unboxed known (Symbol sym)).
  - exact (shown_number_renamed r boxed unboxed known (Const c)).
Qed.

Lemma unbox_arg_renamed (r : renaming) (boxed : boxable_number)
    (unboxed : list variable) (known : facts) (s : simple) :
  unbox_arg boxed (map (rename_variable r) unboxed) (rename_facts r known)
    (rename_simple r s)
  = rename_simple r (unbox_arg boxed unboxed known s).
Proof.
  unfold unbox_arg. rewrite unboxed_simple_renamed.
  destruct (unboxed_simple boxed unboxed known s); reflexivity.
Qed.

Lemma together_args_renamed (r : renaming) (boxed : boxable_number)
    (us : list nat) (unboxed : list variable) (known : facts)
    (args : list simple) :
  together_args boxed us (map (rename_variable r) unboxed)
    (rename_facts r known) (map (rename_simple r) args)
  = map (rename_simple r) (together_args boxed us unboxed known args).
Proof.
  unfold together_args.
  rewrite kept_from_map, moved_from_map, map_app, map_rev, !map_map.
  do 2 f_equal. apply map_ext. intros s. apply unbox_arg_renamed.
Qed.

Lemma together_apply_cont_renamed (r : renaming) (boxed : boxable_number)
    (active : positions) (unboxed : list variable) (known : facts)
    (ac : apply_cont) :
  together_apply_cont boxed (rename_keyed r active)
    (map (rename_variable r) unboxed) (rename_facts r known)
    (rename_apply_cont r ac)
  = rename_apply_cont r (together_apply_cont boxed active unboxed known ac).
Proof.
  unfold together_apply_cont, rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite positions_of_renamed, together_args_renamed. reflexivity.
Qed.

Lemma unbox_of_renamed (r : renaming) (boxed : boxable_number) (d : named) :
  unbox_of boxed (rename_named r d)
  = option_map (rename_variable r) (unbox_of boxed d).
Proof.
  destruct d as [s | [[] | | | | ] [ | [x | sym | c] [ | a args]] | | | | | ];
    try reflexivity.
  cbn [rename_named map rename_simple unbox_of].
  match goal with
  | |- context [kind_eqb ?a ?b] => destruct (kind_eqb a b)
  end; reflexivity.
Qed.

Lemma together_named_renamed (r : renaming) (boxed : boxable_number)
    (unboxed : list variable) (d : named) :
  together_named boxed (map (rename_variable r) unboxed) (rename_named r d)
  = rename_named r (together_named boxed unboxed d).
Proof.
  unfold together_named. rewrite unbox_of_renamed.
  destruct (unbox_of boxed d) as [p | ]; cbn [option_map]; [ | reflexivity ].
  rewrite among_rename_variable. destruct (among p unboxed); reflexivity.
Qed.

Fixpoint together_expr_renamed (r : renaming) (boxed : boxable_number)
    (sites active : positions) (unboxed : list variable) (known : facts)
    (e : expr) {struct e} :
  together_expr boxed (rename_keyed r sites) (rename_keyed r active)
    (map (rename_variable r) unboxed) (rename_facts r known)
    (rename_expr r e)
  = rename_expr r (together_expr boxed sites active unboxed known e)
with together_handler_renamed (r : renaming) (boxed : boxable_number)
    (sites active : positions) (unboxed : list variable) (known : facts)
    (us : list nat) (h : continuation_handler) {struct h} :
  together_handler boxed (rename_keyed r sites) (rename_keyed r active)
    (map (rename_variable r) unboxed) (rename_facts r known) us
    (rename_handler r h)
  = rename_handler r
      (together_handler boxed sites active unboxed known us h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr together_expr]; try reflexivity.
    + rewrite together_named_renamed, without_one_renamed.
      rewrite facts_under_let_equivariant by assumption.
      rewrite together_expr_renamed. reflexivity.
    + rewrite facts_under_statics_equivariant by assumption.
      rewrite together_expr_renamed. reflexivity.
    + rewrite positions_of_renamed, together_handler_renamed.
      rewrite
        <- (together_expr_renamed r boxed sites
              ((k', positions_of sites k') :: active)).
      reflexivity.
    + fold (rename_handlers r handlers).
      rewrite active_under_renamed, together_expr_renamed. f_equal.
      generalize (active_under sites handlers active). intros active'.
      unfold rename_handlers.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map].
      rewrite positions_of_renamed, together_handler_renamed, IH.
      reflexivity.
    + rewrite together_apply_cont_renamed. reflexivity.
    + f_equal. induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map fst snd]. rewrite together_apply_cont_renamed, IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler together_handler].
    rewrite untag_params_list_renamed, untagged_under_renamed.
    rewrite facts_under_params_equivariant by assumption.
    rewrite together_expr_renamed. reflexivity.
Qed.

(* The test. *)

Lemma unboxed_arg_renamed (r : renaming) (boxed : boxable_number)
    (unboxed : list variable) (known : facts) (s : simple) :
  unboxed_arg boxed (map (rename_variable r) unboxed) (rename_facts r known)
    (rename_simple r s)
  = unboxed_arg boxed unboxed known s.
Proof.
  unfold unboxed_arg. rewrite unboxed_simple_renamed.
  destruct (unboxed_simple boxed unboxed known s); reflexivity.
Qed.

Lemma together_args_ok_renamed (r : renaming) (boxed : boxable_number)
    (unboxed : list variable) (known : facts) (us : list nat)
    (args : list simple) :
  together_args_ok boxed (map (rename_variable r) unboxed)
    (rename_facts r known) us (map (rename_simple r) args)
  = together_args_ok boxed unboxed known us args.
Proof.
  unfold together_args_ok. rewrite kept_from_map, moved_from_map. f_equal.
  - apply none_in_renamed. intros p. apply var_in_simples_equivariant.
  - induction (moved_from us 0 args) as [ | s rest IH]; [ reflexivity | ].
    cbn [map forallb]. rewrite unboxed_arg_renamed, IH. reflexivity.
Qed.

Lemma together_apply_cont_ok_renamed (r : renaming) (boxed : boxable_number)
    (active : positions) (unboxed : list variable) (known : facts)
    (ac : apply_cont) :
  together_apply_cont_ok boxed (rename_keyed r active)
    (map (rename_variable r) unboxed) (rename_facts r known)
    (rename_apply_cont r ac)
  = together_apply_cont_ok boxed active unboxed known ac.
Proof.
  unfold together_apply_cont_ok, rename_apply_cont.
  cbn [ac_continuation ac_args].
  rewrite positions_of_renamed. apply together_args_ok_renamed.
Qed.

Lemma reads_unboxed_renamed (r : renaming) (boxed : boxable_number)
    (unboxed : list variable) (d : named) :
  reads_unboxed boxed (map (rename_variable r) unboxed) (rename_named r d)
  = reads_unboxed boxed unboxed d.
Proof.
  unfold reads_unboxed. rewrite unbox_of_renamed.
  destruct (unbox_of boxed d) as [p | ]; cbn [option_map]; [ | reflexivity ].
  apply among_rename_variable.
Qed.

Lemma together_named_ok_renamed (r : renaming) (boxed : boxable_number)
    (unboxed : list variable) (d : named) :
  together_named_ok boxed (map (rename_variable r) unboxed)
    (rename_named r d)
  = together_named_ok boxed unboxed d.
Proof.
  unfold together_named_ok. rewrite reads_unboxed_renamed. f_equal.
  apply none_in_renamed. intros p. apply var_in_named_equivariant.
Qed.

Fixpoint together_ok_renamed (r : renaming) (boxed : boxable_number)
    (sites active : positions) (unboxed : list variable) (known : facts)
    (e : expr) {struct e} :
  together_ok boxed (rename_keyed r sites) (rename_keyed r active)
    (map (rename_variable r) unboxed) (rename_facts r known)
    (rename_expr r e)
  = together_ok boxed sites active unboxed known e
with together_handler_ok_renamed (r : renaming) (boxed : boxable_number)
    (sites active : positions) (unboxed : list variable) (known : facts)
    (us : list nat) (h : continuation_handler) {struct h} :
  together_handler_ok boxed (rename_keyed r sites) (rename_keyed r active)
    (map (rename_variable r) unboxed) (rename_facts r known) us
    (rename_handler r h)
  = together_handler_ok boxed sites active unboxed known us h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr together_ok]; try reflexivity.
    + rewrite together_named_ok_renamed, without_one_renamed.
      rewrite facts_under_let_equivariant by assumption.
      rewrite together_ok_renamed. reflexivity.
    + rewrite facts_under_statics_equivariant by assumption.
      rewrite together_ok_renamed. f_equal.
      apply none_in_renamed. intros p. apply existsb_map_with.
      intros binding. apply var_in_static_binding_equivariant.
    + rewrite positions_of_renamed, together_handler_ok_renamed.
      rewrite
        <- (together_ok_renamed r boxed sites
              ((k', positions_of sites k') :: active)).
      reflexivity.
    + fold (rename_handlers r handlers).
      rewrite active_under_renamed, together_ok_renamed. f_equal.
      generalize (active_under sites handlers active). intros active'.
      unfold rename_handlers.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite positions_of_renamed, together_handler_ok_renamed, IH.
      reflexivity.
    + cbn [rename_apply ap_callee ap_args ap_continuation ap_exn_continuation].
      rewrite !no_positions_renamed. do 2 f_equal.
      apply none_in_renamed. intros p.
      fold (rename_apply r ap). rewrite exn_extra_simples_renamed.
      rewrite var_in_simple_equivariant, !var_in_simples_equivariant.
      reflexivity.
    + apply together_apply_cont_ok_renamed.
    + f_equal.
      * apply none_in_renamed. intros p. apply var_in_simple_equivariant.
      * induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
        cbn [map forallb fst snd].
        rewrite together_apply_cont_ok_renamed, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler together_handler_ok].
    rewrite handler_fits_renamed, untagged_under_renamed.
    rewrite facts_under_params_equivariant by assumption.
    rewrite together_ok_renamed. reflexivity.
Qed.

Lemma together_at_renamed (r : renaming) (boxed : boxable_number)
    (sites : positions) (known : facts) (e : expr) :
  together_at boxed (rename_keyed r sites) (rename_facts r known)
    (rename_expr r e)
  = rename_expr r (together_at boxed sites known e).
Proof.
  unfold together_at.
  rewrite
    (together_ok_renamed r boxed sites [] [] known e
     : together_ok boxed _ [] [] _ _ = _).
  destruct (together_ok boxed sites [] [] known e); [ | reflexivity ].
  exact (together_expr_renamed r boxed sites [] [] known e).
Qed.

(* The positions that the pass uses. *)

Lemma together_args_blame_renamed (r : renaming) (boxed : boxable_number)
    (origin : origin_map) (known : facts) (k : continuation) (us : list nat)
    (args : list simple) :
  ∀ i : nat,
    together_args_blame boxed (rename_origin r origin) (rename_facts r known)
      (rename_continuation r k) us i (map (rename_simple r) args)
    = rename_keyed r (together_args_blame boxed origin known k us i args).
Proof.
  induction args as [ | a args IH]; intros i; [ reflexivity | ].
  cbn [map together_args_blame]. rewrite IH, rename_keyed_app. f_equal.
  destruct (at_position us i).
  - rewrite map_fst_rename_origin, unboxed_arg_renamed.
    destruct (unboxed_arg boxed (map fst origin) known a); reflexivity.
  - apply origins_renamed. intros x. apply var_in_simple_equivariant.
Qed.

Lemma together_apply_cont_blame_renamed (r : renaming)
    (boxed : boxable_number) (active : positions) (origin : origin_map)
    (known : facts) (ac : apply_cont) :
  together_apply_cont_blame boxed (rename_keyed r active)
    (rename_origin r origin) (rename_facts r known) (rename_apply_cont r ac)
  = rename_keyed r (together_apply_cont_blame boxed active origin known ac).
Proof.
  unfold together_apply_cont_blame, rename_apply_cont.
  cbn [ac_continuation ac_args].
  rewrite positions_of_renamed. apply together_args_blame_renamed.
Qed.

Fixpoint together_blame_renamed (r : renaming) (boxed : boxable_number)
    (sites active : positions) (origin : origin_map) (known : facts)
    (e : expr) {struct e} :
  together_blame boxed (rename_keyed r sites) (rename_keyed r active)
    (rename_origin r origin) (rename_facts r known) (rename_expr r e)
  = rename_keyed r (together_blame boxed sites active origin known e)
with together_handler_blame_renamed (r : renaming) (boxed : boxable_number)
    (sites active : positions) (origin : origin_map) (known : facts)
    (k : continuation) (h : continuation_handler) {struct h} :
  together_handler_blame boxed (rename_keyed r sites) (rename_keyed r active)
    (rename_origin r origin) (rename_facts r known) (rename_continuation r k)
    (rename_handler r h)
  = rename_keyed r
      (together_handler_blame boxed sites active origin known k h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr together_blame]; try reflexivity.
    + rewrite map_fst_rename_origin, reads_unboxed_renamed, unbind_renamed.
      rewrite facts_under_let_equivariant by assumption.
      rewrite together_blame_renamed, rename_keyed_app.
      f_equal.
      destruct (reads_unboxed boxed (map fst origin) d); [ reflexivity | ].
      apply origins_renamed. intros x. apply var_in_named_equivariant.
    + rewrite facts_under_statics_equivariant by assumption.
      rewrite together_blame_renamed, rename_keyed_app. f_equal.
      apply origins_renamed. intros x. apply existsb_map_with.
      intros binding. apply var_in_static_binding_equivariant.
    + rewrite positions_of_renamed, together_handler_blame_renamed,
        rename_keyed_app.
      rewrite
        <- (together_blame_renamed r boxed sites
              ((k', positions_of sites k') :: active)).
      reflexivity.
    + fold (rename_handlers r handlers).
      rewrite active_under_renamed, together_blame_renamed, rename_keyed_app.
      f_equal.
      generalize (active_under sites handlers active). intros active'.
      unfold rename_handlers.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map flat_map].
      rewrite together_handler_blame_renamed, IH, rename_keyed_app.
      reflexivity.
    + cbn [rename_apply ap_callee ap_args ap_continuation ap_exn_continuation].
      rewrite !all_of_renamed, !rename_keyed_app. f_equal.
      apply origins_renamed. intros x.
      fold (rename_apply r ap). rewrite exn_extra_simples_renamed.
      rewrite var_in_simple_equivariant, !var_in_simples_equivariant.
      reflexivity.
    + apply together_apply_cont_blame_renamed.
    + rewrite rename_keyed_app. f_equal.
      * apply origins_renamed. intros x. apply var_in_simple_equivariant.
      * induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
        cbn [map flat_map fst snd].
        rewrite together_apply_cont_blame_renamed, IH, rename_keyed_app.
        reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler together_handler_blame].
    rewrite positions_of_renamed, handler_fits_renamed, origins_under_renamed.
    rewrite facts_under_params_equivariant by assumption.
    rewrite together_blame_renamed, rename_keyed_app.
    f_equal.
    destruct (handler_fits (positions_of sites k) params metadata);
      [ reflexivity | ].
    unfold rename_keyed. rewrite map_map. reflexivity.
Qed.

Lemma together_settle_renamed (r : renaming) (boxed : boxable_number)
    (fuel : nat) :
  ∀ (sites : positions) (known : facts) (e : expr),
    together_settle boxed fuel (rename_keyed r sites) (rename_facts r known)
      (rename_expr r e)
    = rename_keyed r (together_settle boxed fuel sites known e).
Proof.
  induction fuel as [ | fuel IH]; intros sites known e; [ reflexivity | ].
  cbn [together_settle].
  rewrite
    (together_blame_renamed r boxed sites [] [] known e
     : together_blame boxed _ [] [] _ _ = _).
  destruct (together_blame boxed sites [] [] known e)
    as [ | refused0 refused] eqn:Hblame; [ reflexivity | ].
  cbn [rename_keyed map].
  exact (eq_trans
    (f_equal
       (fun s =>
          together_settle boxed fuel s (rename_facts r known)
            (rename_expr r e))
       (prune_renamed r (refused0 :: refused) sites))
    (IH (prune (refused0 :: refused) sites) known e)).
Qed.

Lemma together_positions_renamed (r : renaming) (boxed : boxable_number)
    (known : facts) (e : expr) :
  together_positions boxed (rename_facts r known) (rename_expr r e)
  = rename_keyed r (together_positions boxed known e).
Proof.
  unfold together_positions.
  rewrite candidates_renamed, count_positions_renamed.
  apply together_settle_renamed.
Qed.

Lemma unbox_together_of_renamed (r : renaming) (boxed : boxable_number)
    (known : facts) (e : expr) :
  unbox_together_of boxed (rename_facts r known) (rename_expr r e)
  = rename_expr r (unbox_together_of boxed known e).
Proof.
  unfold unbox_together_of.
  rewrite together_positions_renamed. apply together_at_renamed.
Qed.

(* The sorts of box that a term unboxes. *)

Fixpoint unboxed_sorts_renamed (r : renaming) (e : expr) {struct e} :
  unboxed_sorts (rename_expr r e) = unboxed_sorts e
with handler_unboxed_sorts_renamed (r : renaming)
    (h : continuation_handler) {struct h} :
  handler_unboxed_sorts (rename_handler r h) = handler_unboxed_sorts h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr unboxed_sorts]; try reflexivity.
    + rewrite unboxed_sorts_renamed.
      destruct d
        as [s | [[] | | | | ] [ | [x | sym | c] [ | a args]] | | | | | ];
        reflexivity.
    + apply unboxed_sorts_renamed.
    + rewrite handler_unboxed_sorts_renamed, unboxed_sorts_renamed.
      reflexivity.
    + rewrite unboxed_sorts_renamed. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map flat_map]. rewrite handler_unboxed_sorts_renamed, IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_unboxed_sorts].
    apply unboxed_sorts_renamed.
Qed.

Lemma unbox_together_all_renamed (r : renaming)
    (boxes : list boxable_number) :
  ∀ (known : facts) (e : expr),
    unbox_together_all boxes (rename_facts r known) (rename_expr r e)
    = rename_expr r (unbox_together_all boxes known e).
Proof.
  induction boxes as [ | boxed boxes IH]; intros known e;
    cbn [unbox_together_all]; [ reflexivity | ].
  rewrite unbox_together_of_renamed. apply IH.
Qed.

Lemma unbox_together_knowing_renamed (r : renaming) (known : facts)
    (e : expr) :
  unbox_together_knowing (rename_facts r known) (rename_expr r e)
  = rename_expr r (unbox_together_knowing known e).
Proof.
  unfold unbox_together_knowing. rewrite unboxed_sorts_renamed.
  apply unbox_together_all_renamed.
Qed.

End Renamed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

(* A handler whose code is closed, and whose free variables are those
   of another or among the names given. *)
Definition scoped_handler_under (names : list variable)
    (h h' : continuation_handler) : Prop :=
  handler_code_closed h = true ->
  handler_code_closed h' = true
  ∧ incl (free_variables_of_handler h')
      (free_variables_of_handler h ++ names)
  ∧ incl (free_continuations_of_handler h')
      (free_continuations_of_handler h).

Definition scoped_handlers_under (names : list variable)
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
  forallb (fun '(_, h) => handler_code_closed h) handlers' = true
  ∧ incl (flat_map (fun '(_, h) => free_variables_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_variables_of_handler h) handlers
       ++ names)
  ∧ incl
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers).

Lemma scoped_handler_under_intro (names : list variable) (us : list nat)
    (params : list variable) (metadata metadata' : handler_metadata)
    (handler handler' : expr) :
  scoped_under names handler handler' ->
  scoped_handler_under names (Handler params metadata handler)
    (Handler (untag_params_list us params) metadata' handler').
Proof.
  intros Hhandler Hclosed.
  cbn [handler_code_closed free_variables_of_handler
       free_continuations_of_handler] in Hclosed |- *.
  destruct (Hhandler Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros a Ha. apply in_without in Ha as [Ha Hnot].
  apply Hv, in_app_or in Ha. apply in_or_app.
  destruct Ha as [Ha | Ha]; [ left | right; exact Ha ].
  apply in_without. split; [ exact Ha | ].
  intros Hin. apply Hnot. unfold untag_params_list.
  exact (Permutation_in a (kept_moved_permutation us params 0) Hin).
Qed.

Lemma scoped_under_let_cont_handler (names : list variable)
    (k : continuation) (h h' : continuation_handler) (body body' : expr) :
  scoped_handler_under names h h' ->
  scoped_under names body body' ->
  scoped_under names (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  intros Hh Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hh Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  split; [ rewrite Hfirst', Hrest'; reflexivity | ].
  split; [ apply incl_app_under; assumption | ].
  apply incl_app_app; [ exact Hk | apply incl_without, Hk' ].
Qed.

Lemma scoped_under_let_cont_recursive_handlers (names : list variable)
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) :
  map fst handlers' = map fst handlers ->
  scoped_handlers_under names handlers handlers' ->
  scoped_under names body body' ->
  scoped_under names (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers' body').
Proof.
  intros Hnames Hhandlers Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hhandlers Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  split; [ rewrite Hfirst', Hrest'; reflexivity | ].
  split; [ apply incl_app_under; assumption | ].
  rewrite Hnames. apply incl_without, incl_app_app; assumption.
Qed.

Lemma scoped_handlers_under_nil (names : list variable) :
  scoped_handlers_under names [] [].
Proof. intros _. split; [ reflexivity | split; intros a [] ]. Qed.

Lemma scoped_handlers_under_cons (names : list variable)
    (k k' : continuation) (h h' : continuation_handler)
    (handlers handlers' : list (continuation * continuation_handler)) :
  scoped_handler_under names h h' ->
  scoped_handlers_under names handlers handlers' ->
  scoped_handlers_under names ((k, h) :: handlers) ((k', h') :: handlers').
Proof.
  intros Hh Hhandlers Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hh Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hhandlers Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'.
  split; [ reflexivity | split ];
    [ apply incl_app_under; assumption | apply incl_app_app; assumption ].
Qed.

(* The number that the term shows to be in a box is a constant or a
   variable of what is known. *)
Lemma shown_number_free (names : list variable) (boxed : boxable_number)
    (unboxed : list variable) (known : facts) (s number : simple) :
  facts_within names known ->
  shown_number boxed unboxed known s = Some number ->
  incl (free_variables_of_simple number) names.
Proof.
  intros Hknown. unfold shown_number.
  destruct
    (certain_projection known (Unbox_number boxed) (unboxed_kind boxed) s)
    as [projection | ] eqn:Hcertain; [ | discriminate ].
  destruct (none_in unboxed (fun p => var_in_simple p projection));
    [ | discriminate ].
  intros [= <-]. unfold certain_projection in Hcertain.
  assert (Hmade :
    match
      made_field known (Unbox_number boxed) (unboxed_kind boxed) s
    with
    | Some contents => Some contents
    | None => made_number known (Unbox_number boxed) (unboxed_kind boxed) s
    end
    = Some projection ->
    incl (free_variables_of_simple projection) names).
  { destruct (made_field known (Unbox_number boxed) (unboxed_kind boxed) s)
      as [contents | ] eqn:Hfield.
    - intros [= <-].
      exact (made_field_free names known _ _ s contents Hknown Hfield).
    - exact (made_number_free names known _ _ s projection Hknown). }
  destruct (projection_of known (Unbox_number boxed) s)
    as [[y | sym | c] | ]; try exact (Hmade Hcertain).
  destruct (kind_eqb (kind_of_const c) (unboxed_kind boxed));
    [ | discriminate ].
  injection Hcertain as <-. intros z [].
Qed.

Lemma unbox_arg_free (names : list variable) (boxed : boxable_number)
    (unboxed : list variable) (known : facts) (s : simple) :
  facts_within names known ->
  incl (free_variables_of_simple (unbox_arg boxed unboxed known s))
    (free_variables_of_simple s ++ names).
Proof.
  intros Hknown. unfold unbox_arg.
  destruct (unboxed_simple boxed unboxed known s) as [s' | ] eqn:Hs;
    [ | apply incl_appl, incl_refl ].
  assert (Hshown :
    shown_number boxed unboxed known s = Some s' ->
    incl (free_variables_of_simple s') (free_variables_of_simple s ++ names)).
  { intros Hshown. apply incl_appr.
    exact (shown_number_free names boxed unboxed known s s' Hknown Hshown). }
  destruct s as [x | sym | c]; cbn [unboxed_simple] in Hs;
    try exact (Hshown Hs).
  destruct (among x unboxed); [ | exact (Hshown Hs) ].
  injection Hs as <-. apply incl_appl, incl_refl.
Qed.

Lemma together_apply_cont_free_variables (names : list variable)
    (boxed : boxable_number) (active : positions) (unboxed : list variable)
    (known : facts) (ac : apply_cont) :
  facts_within names known ->
  incl
    (free_variables_of_apply_cont
       (together_apply_cont boxed active unboxed known ac))
    (free_variables_of_apply_cont ac ++ names).
Proof.
  intros Hknown.
  unfold free_variables_of_apply_cont, free_variables_of_simples,
    together_apply_cont, together_args.
  cbn [ac_args]. intros a Ha. apply in_flat_map in Ha as (s & Hs & Ha).
  apply in_app_or in Hs as [Hs | Hs].
  - apply in_or_app. left. apply in_flat_map. exists s.
    split; [ exact (in_kept_from _ s _ 0 Hs) | exact Ha ].
  - apply in_rev in Hs. apply in_map_iff in Hs as (s0 & <- & Hs0).
    apply (unbox_arg_free names boxed unboxed known s0 Hknown), in_app_or
      in Ha.
    apply in_or_app. destruct Ha as [Ha | Ha]; [ left | right; exact Ha ].
    apply in_flat_map. exists s0.
    split; [ exact (in_moved_from _ s0 _ 0 Hs0) | exact Ha ].
Qed.

Lemma together_named_free_variables (boxed : boxable_number)
    (unboxed : list variable) (d : named) :
  incl (free_variables_of_named (together_named boxed unboxed d))
    (free_variables_of_named d).
Proof.
  unfold together_named.
  destruct (unbox_of boxed d) as [p | ] eqn:Hunbox; [ | apply incl_refl ].
  destruct (among p unboxed); [ | apply incl_refl ].
  apply FlambdaUnboxTogetherProofs.unbox_of_some in Hunbox. subst d.
  intros a Ha. exact Ha.
Qed.

Fixpoint together_expr_scoped (boxed : boxable_number)
    (sites active : positions) (unboxed : list variable) (e : expr)
    {struct e} :
  ∀ (names : list variable) (known : facts),
    facts_within names known ->
    scoped_under names e (together_expr boxed sites active unboxed known e)
with together_handler_scoped (boxed : boxable_number)
    (sites active : positions) (unboxed : list variable) (us : list nat)
    (h : continuation_handler) {struct h} :
  ∀ (names : list variable) (known : facts),
    facts_within names known ->
    scoped_handler_under names h
      (together_handler boxed sites active unboxed known us h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros names known Hknown; cbn [together_expr];
      try apply scoped_under_refl.
    + apply scoped_under_let.
      * eapply incl_tran;
          [ apply together_named_free_variables | apply incl_appl, incl_refl ].
      * apply together_expr_scoped, facts_under_let_within, Hknown.
    + apply scoped_under_let_static.
      apply together_expr_scoped, facts_under_statics_within, Hknown.
    + apply scoped_under_let_cont_handler;
        [ apply together_handler_scoped, Hknown
        | apply together_expr_scoped, Hknown ].
    + apply scoped_under_let_cont_recursive_handlers;
        [ | | apply together_expr_scoped, Hknown ].
      * rewrite map_map. apply map_ext. intros [k' hh]. reflexivity.
      * generalize (active_under sites handlers active). intros active'.
        induction handlers as [ | [k' hh] handlers IH]; cbn [map];
          [ apply scoped_handlers_under_nil | ].
        apply scoped_handlers_under_cons;
          [ apply together_handler_scoped, Hknown | exact IH ].
    + intros _.
      cbn [code_closed free_variables_of_expr free_continuations_of_expr].
      split; [ reflexivity | split ].
      * apply together_apply_cont_free_variables, Hknown.
      * apply incl_refl.
    + intros _.
      cbn [code_closed free_variables_of_expr free_continuations_of_expr].
      split; [ reflexivity | split ].
      * apply incl_app_under; [ apply incl_appl, incl_refl | ].
        induction arms as [ | [n ac] arms IH]; [ intros a [] | ].
        cbn [map flat_map snd].
        apply incl_app_under;
          [ apply together_apply_cont_free_variables, Hknown | exact IH ].
      * induction arms as [ | [n ac] arms IH]; [ apply incl_refl | ].
        cbn [map flat_map snd].
        apply incl_app_app; [ apply incl_refl | exact IH ].
  - destruct h as [params metadata handler]; intros names known Hknown.
    cbn [together_handler].
    apply scoped_handler_under_intro, together_expr_scoped,
      facts_under_params_within, Hknown.
Qed.

Lemma together_at_scoped (boxed : boxable_number) (sites : positions)
    (names : list variable) (known : facts) (e : expr) :
  facts_within names known ->
  scoped_under names e (together_at boxed sites known e).
Proof.
  intros Hknown. unfold together_at.
  destruct (together_ok boxed sites [] [] known e);
    [ apply together_expr_scoped, Hknown | apply scoped_under_refl ].
Qed.

Lemma unbox_together_all_scoped (boxes : list boxable_number) :
  ∀ (names : list variable) (known : facts) (e : expr),
    facts_within names known ->
    scoped_under names e (unbox_together_all boxes known e).
Proof.
  induction boxes as [ | boxed boxes IH]; intros names known e Hknown;
    cbn [unbox_together_all]; [ apply scoped_under_refl | ].
  eapply scoped_under_trans; [ | apply IH, Hknown ].
  unfold unbox_together_of. apply together_at_scoped, Hknown.
Qed.

Lemma unbox_together_knowing_scoped (names : list variable) (known : facts)
    (e : expr) :
  facts_within names known ->
  scoped_under names e (unbox_together_knowing known e).
Proof.
  intros Hknown. unfold unbox_together_knowing.
  apply unbox_together_all_scoped, Hknown.
Qed.
