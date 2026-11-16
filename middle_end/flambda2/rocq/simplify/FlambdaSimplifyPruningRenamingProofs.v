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


(* Proofs of the statements of FlambdaSimplifyPruningRenamingSpec.v.
   This file needs no review: each statement is closed by a [Theorem]
   whose type is the statement. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNominalSpec
  FlambdaNominalProofs
  FlambdaAlphaSpec FlambdaAlphaProofs FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaValidatorEquivarianceSpec
  FlambdaValidatorEquivarianceBaseProofs FlambdaValidatorEquivarianceProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeWellFormedSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesInvariantSpec FlambdaEnvAliasesRenamingSpec
  FlambdaEnvAliasesRenamingProofs FlambdaTypingEnv
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeet FlambdaMeetSpec FlambdaMeetRenameSpec
  FlambdaMeetRenameProofs FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyPrimitivesRenamingSpec
  FlambdaSimplifyPrimitivesRenamingProofs FlambdaSimplifyApplySpec
  FlambdaSimplifyApplyProofs FlambdaSimplifyStaticsSpec
  FlambdaSimplifyStaticsProofs FlambdaSimplifyCodeEnvSpec
  FlambdaSimplifyCodeEnvProofs FlambdaSimplifyHandlersSpec
  FlambdaSimplifyHandlersProofs FlambdaSimplifyPassSpec
  FlambdaSimplifyPassRenamingSpec FlambdaSimplifyPassRenamingProofs
  FlambdaSimplifyPruningSpec FlambdaSimplifyPruningRenamingSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec
  FlambdaSimplifySwitchSpec FlambdaSimplifySwitchProofs.
From OxCamlFlambda2 Require Import FlambdaNormaliseSpec
  FlambdaUnusedContsSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* [code_env_in] is the function of FlambdaSimplifyCodeEnvSpec.v. The
   theorem of that name in FlambdaSimplifyCodeEnvProofs.v is not used
   here. *)
Local Abbreviation code_env_in :=
  FlambdaSimplifyCodeEnvSpec.code_env_in.

(* ======================================================================== *)
(* A function that rewrites a [Switch]                                      *)
(* ======================================================================== *)

Theorem simplify_switch_rewrite_equivariant
    : simplify_switch_rewrite_equivariant_statement.
Proof.
  intros meet Hmeet r tenv scrutinee arms.
  exact (FlambdaSimplifySwitchProofs.simplify_switch_rename meet r tenv
           scrutinee arms Hmeet).
Qed.

Theorem no_pruning_equivariant : no_pruning_equivariant_statement.
Proof. intros r tenv scrutinee arms. reflexivity. Qed.

(* ======================================================================== *)
(* The test of the types of an environment                                  *)
(* ======================================================================== *)

(* [Keyed.for_all_bindings_b] of a map whose keys are renamed, for a
   renaming of keys that the equality test of keys does not see. *)
Lemma for_all_bindings_b_renamed {K B C : Type} (eqb : K -> K -> bool)
    (f : K -> K) (g : B -> C) (p : K -> B -> bool) (p' : K -> C -> bool)
    (bindings : Keyed.finite_map K B) :
  (∀ k k' : K, eqb (f k) (f k') = eqb k k') ->
  (∀ (k : K) (b : B), p' (f k) (g b) = p k b) ->
  Keyed.for_all_bindings_b eqb p' (Keyed.map_bindings f g bindings)
  = Keyed.for_all_bindings_b eqb p bindings.
Proof.
  intros Heqb Hp. unfold Keyed.for_all_bindings_b, Keyed.map_bindings.
  match goal with
  | |- ?test' [] ?renamed = ?test [] bindings =>
    enough (H : ∀ seen : list K,
              test' (map f seen) renamed = test seen bindings)
      by exact (H [])
  end.
  induction bindings as [ | [k b] bindings IH]; intros seen;
    cbn [map fst snd]; [ reflexivity | ].
  rewrite (among_keys_renamed eqb f Heqb), Hp.
  change (f k :: map f seen) with (map f (k :: seen)). rewrite IH.
  reflexivity.
Qed.

Lemma type_fits_b_renamed (r : renaming) (tenv : Typing_env.t) (n : name)
    (entry : flambda_type * Binding_time.With_name_mode.t) :
  type_fits_b (rename_typing_env r tenv) (rename_name r n)
    (rename_binding r entry)
  = type_fits_b tenv n entry.
Proof.
  unfold type_fits_b, rename_binding. cbn [fst snd].
  rewrite (proj1 type_well_formed_rename),
    (proj1 aliases_with_kinds_rename_type),
    (proj1 equation_names_rename_type), is_obviously_bottom_renamed,
    kind_rename_type.
  rewrite
    (forallb_renamed
       (fun alias : simple * flambda_kind =>
          (rename_simple r (fst alias), snd alias))
       (fun alias => simple_of_kind_b tenv (fst alias) (snd alias)))
    by (intros alias; apply simple_of_kind_b_renamed).
  rewrite
    (forallb_renamed
       (fun equation : name * flambda_kind =>
          (rename_name r (fst equation), snd equation))
       (fun equation =>
          simple_of_kind_b tenv (simple_of_name (fst equation))
            (snd equation)))
    by (intros equation; cbn [fst snd]; rewrite <- rename_simple_of_name;
        apply simple_of_kind_b_renamed).
  destruct n; reflexivity.
Qed.

Theorem types_fit_b_equivariant : types_fit_b_equivariant_statement.
Proof.
  intros r tenv. unfold types_fit_b. rewrite names_to_types_renamed.
  apply for_all_bindings_b_renamed.
  - apply name_eqb_renamed.
  - intros n entry. apply type_fits_b_renamed.
Qed.

(* ======================================================================== *)
(* The tests by which the boolean stays true                                *)
(* ======================================================================== *)

Lemma is_canonical_b_renamed (r : renaming) (tenv : Typing_env.t)
    (e : simple) :
  is_canonical_b (Typing_env.aliases (rename_typing_env r tenv))
    (rename_simple r e)
  = is_canonical_b (Typing_env.aliases tenv) e.
Proof.
  unfold is_canonical_b. rewrite aliases_equivariant, canonical_renamed.
  destruct (Aliases.canonical (Typing_env.aliases tenv) e); reflexivity.
Qed.

Theorem alias_keeps_fit_equivariant : alias_keeps_fit_equivariant_statement.
Proof.
  intros r tenv e1 e2 kind. unfold alias_keeps_fit.
  rewrite !is_canonical_b_renamed, !mem_simple_equivariant,
    !simple_of_kind_b_renamed.
  reflexivity.
Qed.

(* The equations of the levels before the current one. *)
Lemma prev_equations_renamed (r : renaming) (tenv : Typing_env.t) :
  flat_map Typing_env_level.equations
    (map Typing_env.One_level.level
       (Typing_env.prev_levels (rename_typing_env r tenv)))
  = Keyed.map_bindings (rename_name r) (rename_type r)
      (flat_map Typing_env_level.equations
         (map Typing_env.One_level.level (Typing_env.prev_levels tenv))).
Proof.
  cbn [rename_typing_env Typing_env.prev_levels].
  induction (Typing_env.prev_levels tenv) as [ | one_level levels IH];
    cbn [map flat_map]; [ reflexivity | ].
  rewrite IH. unfold Keyed.map_bindings. rewrite map_app. reflexivity.
Qed.

Theorem equation_keeps_fit_equivariant
    : equation_keeps_fit_equivariant_statement.
Proof.
  intros r tenv n ty. unfold equation_keeps_fit.
  rewrite mem_equivariant, (proj1 free_variables_rename_type),
    names_to_types_renamed, lookup_named_renamed, kind_rename_type,
    get_alias_opt_renamed, <- rename_simple_of_name, is_canonical_b_renamed,
    is_obviously_unknown_renamed, prev_equations_renamed,
    lookup_named_renamed, simple_of_kind_b_renamed.
  rewrite
    (forallb_renamed (rename_variable r)
       (fun y => Typing_env.mem None tenv (Name_var y)))
    by (intros y; apply mem_variable_renamed).
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty' binding_time] | ];
    cbn [option_map rename_binding fst];
    [ rewrite kind_rename_type | ];
    (destruct (get_alias_opt ty); cbn [option_map]);
    (destruct
       (Keyed.lookup name_eqb n
          (flat_map Typing_env_level.equations
             (map Typing_env.One_level.level (Typing_env.prev_levels tenv))));
     reflexivity).
Qed.

Theorem let_keeps_fit_equivariant : let_keeps_fit_equivariant_statement.
Proof.
  intros r tenv x d. unfold let_keeps_fit.
  rewrite simplify_named_equivariant.
  destruct (simplify_named tenv d) as [ | s | ty | kind | ];
    cbn [rename_outcome]; try reflexivity.
  - rewrite simplify_simple_equivariant.
    destruct (simplify_simple tenv s) as [[ty canonical] | ];
      cbn [option_map rename_answer fst snd]; [ | reflexivity ].
    rewrite kind_rename_type, add_variable_definition_equivariant.
    change (Var (rename_variable r x)) with (rename_simple r (Var x)).
    apply alias_keeps_fit_equivariant.
  - rewrite kind_rename_type, add_variable_definition_equivariant.
    change (Name_var (rename_variable r x))
      with (rename_name r (Name_var x)).
    apply equation_keeps_fit_equivariant.
Qed.

Lemma block_like_bindings_keep_fit_renamed (r : renaming)
    (bindings : list static_binding) :
  ∀ tenv : Typing_env.t,
    block_like_bindings_keep_fit (rename_typing_env r tenv)
      (map (rename_static_binding r) bindings)
    = block_like_bindings_keep_fit tenv bindings.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; intros tenv;
    cbn [map rename_static_binding block_like_bindings_keep_fit];
    [ reflexivity | apply IH | apply IH | ].
  rewrite is_closure_const_renamed.
  destruct (is_closure_const const); [ apply IH | ].
  rewrite (static_const_type_in_rename r tenv const).
  destruct (static_const_type_in tenv const) as [ty | ]; cbn [option_map].
  - rewrite (add_symbol_definition_equivariant r tenv s).
    change (Name_symbol s) with (rename_name r (Name_symbol s)) at 1.
    rewrite equation_keeps_fit_equivariant. f_equal.
    unfold define_symbol_with_type.
    rewrite (add_symbol_definition_equivariant r tenv s).
    change (Name_symbol s) with (rename_name r (Name_symbol s)) at 1.
    rewrite (replace_equation_equivariant r _ (Name_symbol s) ty).
    apply IH.
  - rewrite (add_symbol_definition_equivariant r tenv s). apply IH.
Qed.

Lemma closure_bindings_keep_fit_renamed (r : renaming)
    (prior : Typing_env.t) (bindings : list static_binding) :
  ∀ tenv : Typing_env.t,
    closure_bindings_keep_fit (rename_typing_env r prior)
      (rename_typing_env r tenv) (map (rename_static_binding r) bindings)
    = closure_bindings_keep_fit prior tenv bindings.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; intros tenv;
    cbn [map rename_static_binding closure_bindings_keep_fit];
    [ reflexivity | apply IH | apply IH | ].
  rewrite is_closure_const_renamed.
  destruct (is_closure_const const); [ | apply IH ].
  rewrite (static_const_type_in_rename r prior const).
  destruct (static_const_type_in prior const) as [ty | ]; cbn [option_map];
    [ | apply IH ].
  change (Name_symbol s) with (rename_name r (Name_symbol s)) at 1 2.
  rewrite equation_keeps_fit_equivariant,
    (replace_equation_equivariant r tenv (Name_symbol s) ty), IH.
  reflexivity.
Qed.

Theorem statics_keep_fit_equivariant
    : statics_keep_fit_equivariant_statement.
Proof.
  intros r tenv bindings. unfold statics_keep_fit. cbv zeta.
  rewrite closure_symbols_of_renamed,
    (add_symbol_definitions_equivariant r tenv (closure_symbols_of bindings)),
    block_like_bindings_keep_fit_renamed, add_block_like_bindings_renamed,
    closure_bindings_keep_fit_renamed.
  reflexivity.
Qed.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

Theorem arm_env_keeps_fit_equivariant
    : arm_env_keeps_fit_equivariant_statement.
Proof.
  intros r tenv scrutinee d. unfold arm_env_keeps_fit.
  rewrite simplify_simple_equivariant.
  destruct (simplify_simple tenv scrutinee) as [[ty [c | sym | const]] | ];
    cbn [option_map rename_answer fst snd rename_simple]; try reflexivity.
  change (Var (rename_variable r c)) with (rename_simple r (Var c)).
  rewrite simple_in_terms_b_renamed, simple_of_kind_b_renamed.
  destruct (simple_in_terms_b tenv (Var c)
            && simple_of_kind_b tenv (Var c) naked_immediate_kind);
    [ | reflexivity ].
  pose proof
    (add_alias_equivariant r tenv (Var c) (Const (Naked_immediate d)))
    as Hadd.
  pose proof
    (alias_keeps_fit_equivariant r tenv (Var c) (Const (Naked_immediate d))
       naked_immediate_kind)
    as Hkeeps.
  cbn [rename_simple] in Hadd, Hkeeps |- *. rewrite Hadd, Hkeeps.
  destruct (Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)))
    as [[ | result | ] | ];
    reflexivity.
Qed.

Lemma bind_params_to_args_keep_fit_renamed (r : renaming)
    (aliased aliased' : simple -> bool)
    (bindings : list (variable * flambda_kind * simple)) :
  (∀ s : simple, aliased' (rename_simple r s) = aliased s) ->
  ∀ tenv : Typing_env.t,
    bind_params_to_args_keep_fit aliased' (rename_typing_env r tenv)
      (map (rename_param_binding r) bindings)
    = bind_params_to_args_keep_fit aliased tenv bindings.
Proof.
  intros Haliased.
  induction bindings as [ | [[x kind] canonical] bindings IH]; intros tenv;
    [ reflexivity | ].
  cbn [map rename_param_binding fst snd bind_params_to_args_keep_fit].
  rewrite typing_env_well_formed_rename, mem_variable_renamed.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x)));
    [ | reflexivity ].
  rewrite Haliased, simple_in_terms_b_renamed, simple_of_kind_b_renamed,
    add_variable_definition_equivariant.
  destruct (aliased canonical && simple_in_terms_b tenv canonical
            && simple_of_kind_b tenv canonical kind);
    [ | apply IH ].
  change (Var (rename_variable r x)) with (rename_simple r (Var x)).
  rewrite add_alias_equivariant, alias_keeps_fit_equivariant.
  destruct (Typing_env.add_alias
              (Typing_env.add_variable_definition tenv x kind
                 Name_mode.normal)
              (Var x) canonical)
    as [[ | result | ] | ];
    cbn [option_map or_unknown_or_bottom_map rename_add_alias_result
           Typing_env.add_alias_result_t];
    [ apply IH | rewrite IH; reflexivity | apply IH | apply IH ].
Qed.

Theorem use_env_keeps_fit_equivariant
    : use_env_keeps_fit_equivariant_statement.
Proof.
  intros r scope tenv params metadata args. unfold use_env_keeps_fit.
  rewrite simplify_simples_equivariant.
  destruct (simplify_simples tenv args) as [answers | ]; cbn [option_map];
    [ | reflexivity ].
  assert (Hcanonicals :
    map snd (map (rename_answer r) answers)
    = map (rename_simple r) (map snd answers)).
  { rewrite !map_map. reflexivity. }
  rewrite Hcanonicals, bindings_renamed.
  apply bind_params_to_args_keep_fit_renamed.
  intros s. rewrite simple_in_terms_b_renamed, not_among_renamed.
  reflexivity.
Qed.

Theorem leaf_uses_keep_fit_equivariant
    : leaf_uses_keep_fit_equivariant_statement.
Proof.
  intros r tenv k e.
  destruct e
    as [x declared d body | bindings body | k' h body | handlers body | ap
        | ac | scrutinee arms | ];
    cbn [rename_expr leaf_uses_keep_fit]; try reflexivity.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map forallb fst snd rename_apply_cont ac_continuation].
  rewrite rename_continuation_eqb, arm_env_keeps_fit_equivariant, IH.
  reflexivity.
Qed.

Theorem let_env_keeps_fit_equivariant
    : let_env_keeps_fit_equivariant_statement.
Proof.
  intros r tenv x declared d. unfold let_env_keeps_fit. cbv zeta.
  rewrite resolve_value_slot_equivariant, simplify_let_equivariant,
    let_keeps_fit_equivariant.
  destruct (simplify_let tenv x (resolve_value_slot tenv declared d))
    as [ | d' tenv' | ];
    reflexivity.
Qed.

Theorem find_use_keeps_fit_equivariant
    : find_use_keeps_fit_equivariant_statement.
Proof.
  intros r scope tenv k h body. revert tenv.
  induction body
    as [x declared d rest IH | bindings rest _ | k' h' rest IH
        | handlers rest _ | ap | ac | scrutinee arms | ];
    intros tenv; cbn [rename_expr find_use_keeps_fit]; try reflexivity.
  - rewrite let_env_equivariant, let_env_keeps_fit_equivariant.
    destruct (let_env tenv x declared d) as [tenv' | ]; cbn [option_map];
      [ | reflexivity ].
    rewrite IH. reflexivity.
  - apply IH.
  - destruct h as [params metadata handler]. cbn [rename_handler].
    change (Apply_cont (rename_apply_cont r ac))
      with (rename_expr r (Apply_cont ac)).
    rewrite leaf_uses_equivariant, leaf_uses_keep_fit_equivariant.
    destruct (leaf_uses tenv k (Apply_cont ac))
      as [[ | [tenv_use args] [ | second others]] | ];
      cbn [option_map map rename_application fst snd]; try reflexivity.
    rewrite use_env_keeps_fit_equivariant. reflexivity.
  - destruct h as [params metadata handler]. cbn [rename_handler].
    change (Switch (rename_simple r scrutinee)
              (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                 arms))
      with (rename_expr r (Switch scrutinee arms)).
    rewrite leaf_uses_equivariant, leaf_uses_keep_fit_equivariant.
    destruct (leaf_uses tenv k (Switch scrutinee arms))
      as [[ | [tenv_use args] [ | second others]] | ];
      cbn [option_map map rename_application fst snd]; try reflexivity.
    rewrite use_env_keeps_fit_equivariant. reflexivity.
Qed.

Fixpoint gather_uses_keeps_fit_equivariant_aux (r : renaming) (body : expr)
    {struct body} :
  ∀ (tenv : Typing_env.t) (k : continuation) (h : continuation_handler),
    gather_uses_keeps_fit (rename_typing_env r tenv)
      (rename_continuation r k) (rename_handler r h) (rename_expr r body)
    = gather_uses_keeps_fit tenv k h body.
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros tenv k h; cbn [rename_expr gather_uses_keeps_fit];
    try reflexivity.
  - rewrite let_env_equivariant, let_env_keeps_fit_equivariant.
    destruct (let_env tenv x declared d) as [tenv' | ]; cbn [option_map];
      [ | reflexivity ].
    rewrite (gather_uses_keeps_fit_equivariant_aux r rest tenv' k h).
    reflexivity.
  - rewrite (gather_uses_keeps_fit_equivariant_aux r rest tenv k h),
      cont_free_in_handler_b_equivariant.
    destruct (cont_free_in_handler_b k h'); [ | reflexivity ].
    destruct h' as [params' metadata' body']. cbn [rename_handler].
    rewrite handler_env_rename.
    destruct (handler_env tenv params' metadata') as [tenv_handler | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite (gather_uses_keeps_fit_equivariant_aux r body' tenv_handler k h).
    reflexivity.
  - change (Switch (rename_simple r scrutinee)
              (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                 arms))
      with (rename_expr r (Switch scrutinee arms)).
    apply leaf_uses_keep_fit_equivariant.
Qed.

Theorem gather_uses_keeps_fit_equivariant
    : gather_uses_keeps_fit_equivariant_statement.
Proof.
  intros r tenv k h body. apply gather_uses_keeps_fit_equivariant_aux.
Qed.

Lemma find_gathered_use_keeps_fit_equivariant (r : renaming)
    (tenv : Typing_env.t) (k : continuation) (h : continuation_handler)
    (body : expr) :
  find_gathered_use_keeps_fit (rename_typing_env r tenv)
    (rename_continuation r k) (rename_handler r h) (rename_expr r body)
  = find_gathered_use_keeps_fit tenv k h body.
Proof.
  destruct h as [params metadata handler].
  unfold find_gathered_use_keeps_fit. cbn [rename_handler].
  change (Handler (map (rename_variable r) params) metadata
            (rename_expr r handler))
    with (rename_handler r (Handler params metadata handler)).
  rewrite gather_uses_equivariant, gather_uses_keeps_fit_equivariant.
  destruct (gather_uses tenv k (Handler params metadata handler) body)
    as [[[uses xs] ks] | ];
    cbn [option_map rename_gathered fst snd]; [ | reflexivity ].
  destruct uses as [ | [tenv_use args] [ | second others]];
    cbn [map rename_application fst snd]; try reflexivity.
  rewrite use_env_keeps_fit_equivariant. reflexivity.
Qed.

Theorem find_one_use_keeps_fit_equivariant
    : find_one_use_keeps_fit_equivariant_statement.
Proof.
  intros r tenv k h body. unfold find_one_use_keeps_fit.
  rewrite find_use_equivariant, find_use_keeps_fit_equivariant,
    find_gathered_use_keeps_fit_equivariant.
  destruct (find_use tenv tenv k h body) as [found | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

Lemma after_pruning_equivariant (r : renaming) (tenv : Typing_env.t)
    (e : expr) :
  after_pruning (rename_typing_env r tenv) (rename_expr r e)
  = rename_expr r (after_pruning tenv e).
Proof.
  destruct e
    as [x declared d body | bindings body | k handler body
        | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr after_pruning]; try reflexivity.
  - rewrite simplify_apply_cont_equivariant. reflexivity.
  - apply simplify_switch_equivariant.
Qed.

Theorem simplify_switch_pruning_equivariant
    : simplify_switch_pruning_equivariant_statement.
Proof.
  intros prune r tenv fit scrutinee arms Hprune.
  unfold simplify_switch_pruning.
  rewrite types_fit_b_equivariant, mem_simple_equivariant.
  destruct (fit && types_fit_b tenv
            && Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee).
  - rewrite Hprune.
    destruct (prune tenv scrutinee arms) as [e | ]; cbn [option_map].
    + apply after_pruning_equivariant.
    + apply simplify_switch_equivariant.
  - apply simplify_switch_equivariant.
Qed.

Fixpoint simplify_expr_pruning_equivariant_aux (prune : switch_rewrite)
    (Hprune : switch_rewrite_equivariant prune) (r : renaming) (e : expr)
    {struct e} :
  ∀ (tenv : Typing_env.t) (fit : bool),
    simplify_expr_pruning prune (rename_typing_env r tenv) fit
      (rename_expr r e)
    = rename_expr r (simplify_expr_pruning prune tenv fit e)
with simplify_handler_pruning_equivariant (prune : switch_rewrite)
    (Hprune : switch_rewrite_equivariant prune) (r : renaming)
    (h : continuation_handler) {struct h} :
  ∀ (recursive : bool) (tenv : Typing_env.t) (fit : bool),
    simplify_handler_pruning prune recursive (rename_typing_env r tenv) fit
      (rename_handler r h)
    = rename_handler r (simplify_handler_pruning prune recursive tenv fit h)
with simplify_static_binding_pruning_equivariant (prune : switch_rewrite)
    (Hprune : switch_rewrite_equivariant prune) (r : renaming)
    (binding : static_binding) {struct binding} :
  ∀ (tenv : Typing_env.t) (fit : bool),
    simplify_static_binding_pruning prune (rename_typing_env r tenv) fit
      (rename_static_binding r binding)
    = rename_static_binding r
        (simplify_static_binding_pruning prune tenv fit binding)
with simplify_code_pruning_equivariant (prune : switch_rewrite)
    (Hprune : switch_rewrite_equivariant prune) (r : renaming) (c : code)
    {struct c} :
  ∀ (tenv : Typing_env.t) (fit : bool),
    simplify_code_pruning prune (rename_typing_env r tenv) fit
      (rename_code r c)
    = rename_code r (simplify_code_pruning prune tenv fit c).
Proof.
  - destruct e
      as [x declared d body | bindings body | k handler body
          | handlers body | ap | ac | scrutinee arms | ];
      intros tenv fit; cbn [rename_expr simplify_expr_pruning].
    + cbv zeta.
      rewrite resolve_value_slot_equivariant, simplify_let_equivariant,
        let_keeps_fit_equivariant.
      destruct (simplify_let tenv x (resolve_value_slot tenv declared d))
        as [ | d' tenv' | ];
        cbn [rename_let_step rename_expr]; try reflexivity.
      * rewrite
          (simplify_expr_pruning_equivariant_aux prune Hprune r body tenv').
        reflexivity.
      * rewrite declare_variable_equivariant.
        destruct (declare_variable tenv x declared) as [tenv' | ];
          cbn [option_map rename_expr]; [ | reflexivity ].
        rewrite
          (simplify_expr_pruning_equivariant_aux prune Hprune r body tenv').
        reflexivity.
    + cbv zeta.
      rewrite add_static_bindings_rename, statics_keep_fit_equivariant.
      rewrite (simplify_expr_pruning_equivariant_aux prune Hprune r body).
      f_equal.
      generalize (add_static_bindings tenv bindings).
      generalize (fit && statics_keep_fit tenv bindings).
      intros fit' tenv'.
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ reflexivity | ].
      rewrite
        (simplify_static_binding_pruning_equivariant prune Hprune r binding
           tenv' fit'),
        IH.
      reflexivity.
    + pose proof
        (simplify_handler_pruning_equivariant prune Hprune r handler false
           tenv fit)
        as Hdefault.
      rewrite
        (simplify_expr_pruning_equivariant_aux prune Hprune r body tenv fit).
      destruct handler as [params metadata handler].
      cbn [rename_handler].
      change (Handler (map (rename_variable r) params) metadata
                (rename_expr r handler))
        with (rename_handler r (Handler params metadata handler)).
      rewrite find_one_use_equivariant, find_one_use_keeps_fit_equivariant.
      destruct (find_one_use tenv k (Handler params metadata handler)
                  (simplify_expr_pruning prune tenv fit body))
        as [[[target xs] ks] | ];
        cbn [option_map rename_use fst snd];
        [ | rewrite Hdefault; reflexivity ].
      rewrite
        (simplify_expr_pruning_equivariant_aux prune Hprune r handler
           target).
      match goal with
      | |- context [Handler _ _ (rename_expr r ?e')] =>
        change (Handler (map (rename_variable r) params) metadata
                  (rename_expr r e'))
          with (rename_handler r (Handler params metadata e'))
      end.
      rewrite binders_not_free_equivariant.
      match goal with
      | |- context [binders_not_free xs ks ?h'] =>
        destruct (binders_not_free xs ks h')
      end;
        [ reflexivity | ].
      rewrite Hdefault. reflexivity.
    + rewrite (simplify_expr_pruning_equivariant_aux prune Hprune r body).
      f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map];
        [ reflexivity | ].
      rewrite
        (simplify_handler_pruning_equivariant prune Hprune r h true tenv
           fit),
        IH.
      reflexivity.
    + apply simplify_apply_rename.
    + rewrite simplify_apply_cont_equivariant. reflexivity.
    + exact (simplify_switch_pruning_equivariant prune r tenv fit scrutinee
               arms Hprune).
    + reflexivity.
  - destruct h as [params metadata handler]; intros recursive tenv fit;
      cbn [rename_handler simplify_handler_pruning].
    rewrite handler_env_rename.
    destruct (handler_env tenv params metadata) as [tenv_handler | ];
      cbn [option_map]; [ | reflexivity ].
    cbv zeta.
    rewrite
      (simplify_expr_pruning_equivariant_aux prune Hprune r handler
         tenv_handler fit),
      !rename_binds_statics.
    destruct (negb recursive
              || Bool.eqb (binds_statics handler)
                   (binds_statics
                      (simplify_expr_pruning prune tenv_handler fit
                         handler)));
      reflexivity.
  - destruct binding as [id c | id | sym const]; intros tenv fit;
      cbn [rename_static_binding simplify_static_binding_pruning];
      [ rewrite
          (simplify_code_pruning_equivariant prune Hprune r c tenv fit)
      | | ];
      reflexivity.
  - destruct c as [params my_closure ret exn metadata body];
      intros tenv fit; cbn [rename_code simplify_code_pruning].
    rewrite code_env_in_rename.
    destruct (code_env_in tenv params my_closure metadata)
      as [tenv_code | ];
      cbn [option_map]; [ | reflexivity ].
    cbv zeta.
    rewrite
      (simplify_expr_pruning_equivariant_aux prune Hprune r body tenv_code
         fit),
      !rename_binds_statics.
    destruct (negb (binds_statics body)
              && negb
                   (binds_statics
                      (simplify_expr_pruning prune tenv_code fit body)));
      reflexivity.
Qed.

Theorem simplify_expr_pruning_equivariant
    : simplify_expr_pruning_equivariant_statement.
Proof.
  intros prune r tenv fit e Hprune.
  exact (simplify_expr_pruning_equivariant_aux prune Hprune r e tenv fit).
Qed.

Theorem simplify_by_types_pruning_equivariant
    : simplify_by_types_pruning_equivariant_statement.
Proof.
  intros prune Hprune r e. unfold simplify_by_types_pruning.
  rewrite <- (create_equivariant r) at 1.
  apply simplify_expr_pruning_equivariant. exact Hprune.
Qed.

Theorem simplify_by_types_pruning_with_equivariant
    : simplify_by_types_pruning_with_equivariant_statement.
Proof.
  intros join fuel Hjoin. unfold simplify_by_types_pruning_with.
  apply simplify_by_types_pruning_equivariant.
  apply simplify_switch_rewrite_equivariant.
  exact (meet_to_prove_equivariant join fuel Hjoin).
Qed.

Theorem prune_by_types_equivariant : prune_by_types_equivariant_statement.
Proof.
  intros fuel r e. unfold prune_by_types.
  rewrite sets_well_formed_equivariant.
  destruct (sets_well_formed e); [ | reflexivity ].
  apply simplify_by_types_pruning_with_equivariant.
  exact forgetful_join_equivariant.
Qed.
