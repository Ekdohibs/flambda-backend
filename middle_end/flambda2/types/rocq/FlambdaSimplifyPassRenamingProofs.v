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


(* Proofs of the statements of FlambdaSimplifyPassRenamingSpec.v. This
   file needs no review: each statement is closed by a [Theorem] whose
   type is the statement. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaSpec FlambdaAlphaProofs FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaNormaliseSpec FlambdaUnusedContsSpec
  FlambdaValidatorEquivarianceSpec FlambdaValidatorEquivarianceBaseProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeWellFormedSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaTypingEnvInvariantSpec FlambdaTypeRenameSpec
  FlambdaTypeRenameProofs FlambdaEnvRenameSpec FlambdaEnvRenameProofs
  FlambdaProvers FlambdaProversSpec FlambdaProversRenameSpec
  FlambdaProversRenameProofs FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyPrimitivesRenamingSpec
  FlambdaSimplifyPrimitivesRenamingProofs
  FlambdaSimplifyProjectionsSpec FlambdaSimplifyBoxingSpec
  FlambdaSimplifyArithSpec FlambdaSimplifyApplySpec
  FlambdaSimplifyApplyProofs FlambdaSimplifyValueSlotsSpec
  FlambdaSimplifyValueSlotsProofs FlambdaSimplifyStaticsSpec
  FlambdaSimplifyStaticsProofs FlambdaSimplifyCodeEnvSpec
  FlambdaSimplifyCodeEnvProofs FlambdaSimplifyHandlersSpec
  FlambdaSimplifyHandlersProofs FlambdaSimplifyPassSpec
  FlambdaSimplifyPassRenamingSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* [code_env_in] is the function of FlambdaSimplifyCodeEnvSpec.v. The
   theorem of that name in FlambdaSimplifyCodeEnvProofs.v is not used
   here. *)
Local Abbreviation code_env_in :=
  FlambdaSimplifyCodeEnvSpec.code_env_in.

(* An equation of the provers, with the redexes of its statement
   reduced, used from left to right. *)
Ltac rewrite_law law :=
  let E := fresh "E" in
  pose proof law as E; cbv beta in E; rewrite E; clear E.

(* ======================================================================== *)
(* The kinds of renamed types                                               *)
(* ======================================================================== *)

Lemma kinds_renamed (r : renaming) (tys : list flambda_type) :
  map FlambdaTypeGrammar.kind (map (rename_type r) tys)
  = map FlambdaTypeGrammar.kind tys.
Proof.
  rewrite map_map. apply map_ext. intros ty. apply kind_rename_type.
Qed.

(* ======================================================================== *)
(* The outcomes                                                             *)
(* ======================================================================== *)

Lemma of_bool_shortcut_renamed (r : renaming)
    (answer : option (Meet_shortcut.t bool)) (otherwise : outcome) :
  of_bool_shortcut answer (rename_outcome r otherwise)
  = rename_outcome r (of_bool_shortcut answer otherwise).
Proof. destruct answer as [[[ | ] | | ] | ]; reflexivity. Qed.

Lemma of_simple_shortcut_renamed (r : renaming)
    (answer : option (Meet_shortcut.t simple)) (otherwise : outcome) :
  of_simple_shortcut
    (option_map (meet_shortcut_map (rename_simple r)) answer)
    (rename_outcome r otherwise)
  = rename_outcome r (of_simple_shortcut answer otherwise).
Proof. destruct answer as [[s | | ] | ]; reflexivity. Qed.

(* ======================================================================== *)
(* Unary primitives                                                         *)
(* ======================================================================== *)

Lemma simplify_unary_equivariant (r : renaming) (tenv : Typing_env.t)
    (p : unary_primitive) (arg : simple) (ty : flambda_type) :
  simplify_unary (rename_typing_env r tenv) p (rename_simple r arg)
    (rename_type r ty)
  = rename_outcome r (simplify_unary tenv p arg ty).
Proof.
  (* [Tag_immediate] and [Box_number], whose types are made of the type
     of the argument, and the primitives that have no line of their own
     are closed by computation. *)
  destruct p; cbn [simplify_unary]; try reflexivity.
  - (* [Block_load] *)
    match goal with
    | |- context [Block_load _ ?mut ?field] =>
      destruct mut; try reflexivity;
        rewrite_law
          (meet_block_field_simple_rename Name_mode.normal Value
             (Z.to_nat field) r tenv ty);
        apply (of_simple_shortcut_renamed r _ (Stays_with_kind _))
    end.
  - (* [Is_int] *)
    match goal with
    | |- context [Is_int ?variant_only] => destruct variant_only
    end.
    + rewrite_law (meet_is_int_variant_only_rename r tenv ty).
      apply (of_bool_shortcut_renamed r _ (Stays_with_kind _)).
    + rewrite_law (prove_is_int_rename r tenv ty).
      destruct (Provers.prove_is_int tenv ty) as [[[ | ] | ] | ]; reflexivity.
  - (* [Is_null] *)
    rewrite_law (meet_is_null_rename r tenv ty).
    apply (of_bool_shortcut_renamed r _ (Stays_with_kind _)).
  - (* [Get_tag] *)
    rewrite_law (prove_is_int_rename r tenv ty).
    rewrite_law (prove_get_tag_rename r tenv ty).
    destruct (Provers.prove_is_int tenv ty) as [[[ | ] | ] | ];
      try reflexivity;
      destruct (Provers.prove_get_tag tenv ty)
        as [[[ | tag [ | tag' tags]] | ] | ];
      reflexivity.
  - (* [Unbox_number] *)
    rewrite <- rename_type_alias_type_of.
    match goal with
    | |- context [Unbox_number ?kind] =>
      rewrite_law
        (meet_boxed_number_containing_simple_rename kind Name_mode.normal r
           tenv (alias_type_of Value arg))
    end.
    apply (of_simple_shortcut_renamed r _ (Stays_with_kind _)).
  - (* [Untag_immediate] *)
    rewrite types_well_formed_b_rename.
    destruct (types_well_formed_b tenv); [ | reflexivity ].
    rewrite <- rename_type_alias_type_of.
    rewrite_law
      (meet_tagging_of_simple_rename Name_mode.normal r tenv
         (alias_type_of Value arg)).
    apply (of_simple_shortcut_renamed r _ (Stays_with_kind _)).
Qed.

(* ======================================================================== *)
(* The arithmetic of tagged immediates                                      *)
(* ======================================================================== *)

Lemma check_possible_results_renamed (r : renaming)
    (results : list possible_result) (other_side : simple) :
  check_possible_results results (rename_simple r other_side)
  = rename_outcome r (check_possible_results results other_side).
Proof.
  unfold check_possible_results.
  destruct results as [ | result rest]; [ reflexivity | ].
  destruct (forallb (possible_result_eqb result) rest);
    [ destruct result; reflexivity | ].
  match goal with
  | |- context [if ?test then _ else _] => destruct test
  end;
    [ | reflexivity ].
  cbn [rename_outcome]. f_equal. symmetry.
  apply rename_type_these_tagged_immediates.
Qed.

Lemma simplify_int_arith_tagged_equivariant (r : renaming)
    (tenv : Typing_env.t) (op : binary_int_arith_op) (arg1 arg2 : simple)
    (ty1 ty2 : flambda_type) :
  simplify_int_arith_tagged (rename_typing_env r tenv) op
    (rename_simple r arg1) (rename_type r ty1) (rename_simple r arg2)
    (rename_type r ty2)
  = rename_outcome r (simplify_int_arith_tagged tenv op arg1 ty1 arg2 ty2).
Proof.
  assert (Hunknown :
    Stays_with_type any_tagged_immediate
    = rename_outcome r (Stays_with_type any_tagged_immediate)).
  { reflexivity. }
  unfold simplify_int_arith_tagged.
  rewrite !well_formed_input_b_rename.
  rewrite_law (meet_equals_tagged_immediates_rename r tenv ty1).
  rewrite_law (meet_equals_tagged_immediates_rename r tenv ty2).
  destruct (well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2);
    [ | exact Hunknown ].
  destruct (Provers.meet_equals_tagged_immediates tenv ty1)
    as [[nums1 | | ] | ],
    (Provers.meet_equals_tagged_immediates tenv ty2) as [[nums2 | | ] | ];
    try exact Hunknown; try reflexivity;
    repeat match goal with
           | |- context [match ?e with _ => _ end] => destruct e
           end;
    first
      [ exact Hunknown | reflexivity | apply check_possible_results_renamed ].
Qed.

Lemma comparison_unknown_renamed (r : renaming)
    (behaviour : int_comparison_behaviour) :
  rename_outcome r (comparison_unknown behaviour)
  = comparison_unknown behaviour.
Proof. destruct behaviour; reflexivity. Qed.

Lemma simplify_int_comp_tagged_equivariant (r : renaming)
    (tenv : Typing_env.t) (behaviour : int_comparison_behaviour)
    (ty1 ty2 : flambda_type) :
  simplify_int_comp_tagged (rename_typing_env r tenv) behaviour
    (rename_type r ty1) (rename_type r ty2)
  = rename_outcome r (simplify_int_comp_tagged tenv behaviour ty1 ty2).
Proof.
  unfold simplify_int_comp_tagged.
  rewrite !well_formed_input_b_rename.
  rewrite_law (meet_equals_tagged_immediates_rename r tenv ty1).
  rewrite_law (meet_equals_tagged_immediates_rename r tenv ty2).
  repeat match goal with
         | |- context [match ?e with _ => _ end] => destruct e
         end;
    try reflexivity; symmetry; apply comparison_unknown_renamed.
Qed.

Lemma simplify_phys_equal_equivariant (r : renaming) (tenv : Typing_env.t)
    (op : equality_comparison) (ty1 ty2 : flambda_type) :
  simplify_phys_equal (rename_typing_env r tenv) op (rename_type r ty1)
    (rename_type r ty2)
  = rename_outcome r (simplify_phys_equal tenv op ty1 ty2).
Proof.
  unfold simplify_phys_equal.
  rewrite !well_formed_input_b_rename, prove_physical_equality_rename.
  repeat match goal with
         | |- context [match ?e with _ => _ end] => destruct e
         end;
    reflexivity.
Qed.

Lemma simplify_binary_equivariant (r : renaming) (tenv : Typing_env.t)
    (p : binary_primitive) (arg1 arg2 : simple) (ty1 ty2 : flambda_type) :
  simplify_binary (rename_typing_env r tenv) p (rename_simple r arg1)
    (rename_type r ty1) (rename_simple r arg2) (rename_type r ty2)
  = rename_outcome r (simplify_binary tenv p arg1 ty1 arg2 ty2).
Proof.
  destruct p; try reflexivity;
    try apply simplify_phys_equal_equivariant;
    match goal with
    | |- context [Int_arith ?kind _] => destruct kind
    | |- context [Int_comp ?kind _] => destruct kind
    end;
    try reflexivity;
    try (symmetry; apply comparison_unknown_renamed).
  - apply simplify_int_arith_tagged_equivariant.
  - apply simplify_int_comp_tagged_equivariant.
Qed.

Lemma simplify_variadic_equivariant (r : renaming) (p : variadic_primitive)
    (tys : list flambda_type) :
  simplify_variadic p (map (rename_type r) tys)
  = rename_outcome r (simplify_variadic p tys).
Proof.
  destruct p as [[tag] | kind]; cbn [simplify_variadic]; [ | reflexivity ].
  destruct (tag <? no_scan_tag); [ | reflexivity ].
  cbn [rename_outcome]. f_equal. unfold make_block_type. symmetry.
  exact (proj2 (proj2 (proj2
           (rename_type_of_blocks r tag 0%nat (Scannable Value_only) tys
              Alloc_mode.Heap)))
           false).
Qed.

(* ======================================================================== *)
(* A defining expression                                                    *)
(* ======================================================================== *)

(* The value slots of a set of closures, renamed: the same slots, and
   the [Simple]s renamed. *)
Lemma value_slots_renamed (r : renaming)
    (value_slots : list (value_slot * simple)) :
  map fst
    (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
  = map fst value_slots
  ∧ map snd
      (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
    = map (rename_simple r) (map snd value_slots).
Proof. rewrite !map_map. split; reflexivity. Qed.

Theorem simplify_named_equivariant : simplify_named_equivariant_statement.
Proof.
  intros r tenv d. unfold simplify_named.
  (* A renaming leaves the primitive: what was of the heap is. *)
  assert (Hheap :
    FlambdaHeapSpec.named_changes_heap (rename_named r d)
    = FlambdaHeapSpec.named_changes_heap d).
  { destruct d; reflexivity. }
  rewrite Hheap.
  destruct (FlambdaHeapSpec.named_changes_heap d); [ reflexivity | ].
  destruct d
    as [s | p args | code value_slots | slot function_decls value_slots
        | slot closure | project_from slot closure
        | move_from move_to closure];
    cbn [rename_named]; cbn beta iota zeta; try reflexivity;
    try (destruct (value_slots_renamed r value_slots) as [Hslots Hsimples];
         rewrite Hslots, Hsimples, simplify_simples_equivariant;
         destruct (simplify_simples tenv (map snd value_slots))
           as [answers | ];
         cbn [option_map rename_outcome]; [ | reflexivity ];
         rewrite map_fst_rename_answer; unfold set_of_closures_type;
         rewrite closure_of_set_type_rename; reflexivity).
  - rewrite simplify_simples_equivariant.
    destruct (simplify_simples tenv args) as [answers | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite map_fst_rename_answer, kinds_renamed.
    destruct (prim_args_well_kinded p
                (map FlambdaTypeGrammar.kind (map fst answers)));
      [ | reflexivity ].
    destruct p;
      [ destruct args as [ | arg [ | arg' args]]; cbn [map];
        try reflexivity;
        destruct (map fst answers) as [ | ty [ | ty' tys]]; cbn [map];
        try reflexivity;
        apply simplify_unary_equivariant
      | destruct args as [ | arg1 [ | arg2 [ | arg3 args]]]; cbn [map];
        try reflexivity;
        destruct (map fst answers) as [ | ty1 [ | ty2 [ | ty3 tys]]];
        cbn [map]; try reflexivity;
        apply simplify_binary_equivariant
      | destruct args; destruct (map fst answers);
        apply simplify_variadic_equivariant
      | .. ];
      destruct args; destruct (map fst answers); reflexivity.
  - rewrite simplify_simple_equivariant.
    destruct (simplify_simple tenv closure) as [[ty canonical] | ];
      cbn [option_map rename_answer fst]; [ | reflexivity ].
    rewrite_law
      (meet_project_function_slot_simple_rename Name_mode.normal move_to r
         tenv ty).
    apply (of_simple_shortcut_renamed r _ (Stays_with_kind Value)).
Qed.

Lemma canonical_named_equivariant (r : renaming) (tenv : Typing_env.t)
    (d : named) :
  canonical_named (rename_typing_env r tenv) (rename_named r d)
  = rename_named r (canonical_named tenv d).
Proof.
  destruct d; cbn [rename_named canonical_named]; try reflexivity.
  rewrite simplify_simples_equivariant.
  destruct (simplify_simples tenv args) as [answers | ]; cbn [option_map];
    [ | reflexivity ].
  cbn [rename_named]. rewrite map_snd_rename_answer. reflexivity.
Qed.

(* ======================================================================== *)
(* A [Let]                                                                  *)
(* ======================================================================== *)

Lemma mem_variable_renamed (r : renaming) (min_name_mode : option Name_mode.t)
    (tenv : Typing_env.t) (x : variable) :
  Typing_env.mem min_name_mode (rename_typing_env r tenv)
    (Name_var (rename_variable r x))
  = Typing_env.mem min_name_mode tenv (Name_var x).
Proof. exact (mem_equivariant r min_name_mode tenv (Name_var x)). Qed.

Lemma simple_in_terms_b_renamed (r : renaming) (tenv : Typing_env.t)
    (s : simple) :
  simple_in_terms_b (rename_typing_env r tenv) (rename_simple r s)
  = simple_in_terms_b tenv s.
Proof.
  destruct s; cbn [rename_simple simple_in_terms_b];
    [ apply mem_variable_renamed | reflexivity | reflexivity ].
Qed.

Lemma simple_of_kind_b_renamed (r : renaming) (tenv : Typing_env.t)
    (s : simple) (kind : flambda_kind) :
  simple_of_kind_b (rename_typing_env r tenv) (rename_simple r s) kind
  = simple_of_kind_b tenv s kind.
Proof.
  destruct s as [x | sym | c]; cbn [rename_simple simple_of_kind_b].
  - change (Name_var (rename_variable r x))
      with (rename_name r (Name_var x)).
    rewrite find_equivariant.
    destruct (Typing_env.find tenv (Name_var x) (Some kind)); reflexivity.
  - change (Name_symbol sym) with (rename_name r (Name_symbol sym)) at 1.
    rewrite find_equivariant.
    destruct (Typing_env.find tenv (Name_symbol sym) (Some kind));
      reflexivity.
  - reflexivity.
Qed.

Lemma type_in_scope_b_renamed (r : renaming) (tenv : Typing_env.t)
    (ty : flambda_type) :
  type_in_scope_b (rename_typing_env r tenv) (rename_type r ty)
  = type_in_scope_b tenv ty.
Proof.
  unfold type_in_scope_b. rewrite (proj1 free_variables_rename_type).
  induction (free_variables ty) as [ | y ys IH]; cbn [map forallb];
    [ reflexivity | ].
  rewrite IH, mem_variable_renamed. reflexivity.
Qed.

Lemma bind_simple_equivariant (r : renaming) (tenv : Typing_env.t)
    (x : variable) (s : simple) :
  bind_simple (rename_typing_env r tenv) (rename_variable r x)
    (rename_simple r s)
  = rename_let_step r (bind_simple tenv x s).
Proof.
  unfold bind_simple. rewrite simplify_simple_equivariant.
  destruct (simplify_simple tenv s) as [[ty canonical] | ];
    cbn [option_map rename_answer fst snd]; [ | reflexivity ].
  rewrite kind_rename_type, simple_in_terms_b_renamed,
    simple_of_kind_b_renamed.
  destruct (simple_in_terms_b tenv canonical
            && simple_of_kind_b tenv canonical (FlambdaTypeGrammar.kind ty));
    [ | reflexivity ].
  rewrite add_variable_definition_equivariant.
  change (Var (rename_variable r x)) with (rename_simple r (Var x)).
  rewrite add_alias_equivariant.
  destruct (Typing_env.add_alias
              (Typing_env.add_variable_definition tenv x
                 (FlambdaTypeGrammar.kind ty) Name_mode.normal)
              (Var x) canonical)
    as [[ | result | ] | ];
    reflexivity.
Qed.

Theorem simplify_let_equivariant : simplify_let_equivariant_statement.
Proof.
  intros r tenv x d. unfold simplify_let.
  rewrite typing_env_well_formed_rename, mem_variable_renamed.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x)));
    [ | reflexivity ].
  rewrite simplify_named_equivariant.
  destruct (simplify_named tenv d) as [ | s | ty | kind | ];
    cbn [rename_outcome rename_let_step].
  - reflexivity.
  - apply bind_simple_equivariant.
  - rewrite type_in_scope_b_renamed.
    destruct (type_in_scope_b tenv ty); [ | reflexivity ].
    cbn [rename_let_step].
    rewrite canonical_named_equivariant, kind_rename_type,
      add_variable_definition_equivariant.
    change (Name_var (rename_variable r x))
      with (rename_name r (Name_var x)).
    rewrite replace_equation_equivariant. reflexivity.
  - rewrite canonical_named_equivariant, add_variable_definition_equivariant.
    reflexivity.
  - reflexivity.
Qed.

Theorem declare_variable_equivariant
    : declare_variable_equivariant_statement.
Proof.
  intros r tenv x declared. unfold declare_variable.
  rewrite typing_env_well_formed_rename, mem_variable_renamed.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x)));
    [ | reflexivity ].
  cbn [option_map]. rewrite add_variable_definition_equivariant.
  reflexivity.
Qed.

Theorem resolve_value_slot_equivariant
    : resolve_value_slot_equivariant_statement.
Proof.
  intros r tenv declared d.
  destruct d
    as [s | p args | code value_slots | slot function_decls value_slots
        | slot closure | project_from slot closure
        | move_from move_to closure];
    cbn [rename_named resolve_value_slot]; try reflexivity;
    rewrite simplify_simple_equivariant;
    (destruct (simplify_simple tenv closure) as [[ty canonical] | ];
     cbn [option_map rename_answer fst]; [ | reflexivity ]);
    rewrite simplify_project_value_slot_rename;
    (destruct (simplify_project_value_slot tenv declared slot ty) as [s | ];
     reflexivity).
Qed.

(* ======================================================================== *)
(* The other expressions                                                    *)
(* ======================================================================== *)

Theorem simplify_apply_cont_equivariant
    : simplify_apply_cont_equivariant_statement.
Proof.
  intros r tenv [k args action]. unfold simplify_apply_cont.
  cbn [rename_apply_cont ac_args ac_continuation ac_trap_action].
  rewrite simplify_simples_equivariant.
  destruct (simplify_simples tenv args) as [answers | ]; cbn [option_map];
    [ | reflexivity ].
  unfold rename_apply_cont.
  cbn [ac_args ac_continuation ac_trap_action].
  rewrite map_snd_rename_answer. reflexivity.
Qed.

Theorem simplify_switch_equivariant : simplify_switch_equivariant_statement.
Proof.
  intros r tenv scrutinee arms. unfold simplify_switch.
  rewrite simplify_simple_equivariant.
  destruct (simplify_simple tenv scrutinee) as [[ty canonical] | ];
    cbn [option_map rename_answer snd]; [ | reflexivity ].
  destruct canonical as [x | sym | c]; cbn [rename_simple];
    try reflexivity.
  destruct c; try reflexivity.
  rewrite find_arm_renamed.
  match goal with
  | |- context [option_map _ ?found] => destruct found as [ac | ]
  end;
    cbn [option_map]; [ | reflexivity ].
  cbn [rename_expr]. rewrite simplify_apply_cont_equivariant. reflexivity.
Qed.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

Theorem arm_env_equivariant : arm_env_equivariant_statement.
Proof.
  intros r tenv scrutinee d. unfold arm_env.
  rewrite simplify_simple_equivariant.
  destruct (simplify_simple tenv scrutinee) as [[ty [c | sym | const]] | ];
    cbn [option_map rename_answer fst snd rename_simple]; try reflexivity.
  change (Var (rename_variable r c)) with (rename_simple r (Var c)).
  rewrite simple_in_terms_b_renamed, simple_of_kind_b_renamed.
  destruct (simple_in_terms_b tenv (Var c)
            && simple_of_kind_b tenv (Var c) naked_immediate_kind);
    [ | reflexivity ].
  change (Const (Naked_immediate d))
    with (rename_simple r (Const (Naked_immediate d))) at 1.
  rewrite add_alias_equivariant.
  destruct (Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)))
    as [[ | result | ] | ];
    reflexivity.
Qed.

Lemma not_among_renamed (r : renaming) (xs : list variable) (s : simple) :
  not_among (map (rename_variable r) xs) (rename_simple r s)
  = not_among xs s.
Proof.
  destruct s as [x | sym | c]; cbn [rename_simple not_among];
    [ rewrite among_rename_variable | | ]; reflexivity.
Qed.

(* A binding of a parameter, renamed. *)
Definition rename_param_binding (r : renaming)
    (binding : variable * flambda_kind * simple)
    : variable * flambda_kind * simple :=
  (rename_variable r (fst (fst binding)), snd (fst binding),
   rename_simple r (snd binding)).

Lemma bind_params_to_args_equivariant (r : renaming)
    (aliased aliased' : simple -> bool)
    (bindings : list (variable * flambda_kind * simple)) :
  (∀ s : simple, aliased' (rename_simple r s) = aliased s) ->
  ∀ tenv : Typing_env.t,
    bind_params_to_args aliased' (rename_typing_env r tenv)
      (map (rename_param_binding r) bindings)
    = option_map (rename_typing_env r)
        (bind_params_to_args aliased tenv bindings).
Proof.
  intros Haliased.
  induction bindings as [ | [[x kind] canonical] bindings IH]; intros tenv;
    [ reflexivity | ].
  cbn [map rename_param_binding fst snd bind_params_to_args].
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
  rewrite add_alias_equivariant.
  destruct (Typing_env.add_alias
              (Typing_env.add_variable_definition tenv x kind
                 Name_mode.normal)
              (Var x) canonical)
    as [[ | result | ] | ];
    cbn [option_map or_unknown_or_bottom_map rename_add_alias_result
           Typing_env.add_alias_result_t];
    apply IH.
Qed.

Lemma bindings_renamed (r : renaming) (params : list variable) :
  ∀ (kinds : list flambda_kind) (canonicals : list simple),
    combine (combine (map (rename_variable r) params) kinds)
      (map (rename_simple r) canonicals)
    = map (rename_param_binding r) (combine (combine params kinds) canonicals).
Proof.
  induction params as [ | x params IH];
    intros [ | kind kinds] [ | canonical canonicals]; try reflexivity.
  cbn [map combine]. rewrite IH. reflexivity.
Qed.

Theorem use_env_equivariant : use_env_equivariant_statement.
Proof.
  intros r scope tenv params metadata args. unfold use_env.
  rewrite simplify_simples_equivariant.
  destruct (simplify_simples tenv args) as [answers | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite !length_map.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params)
            && Nat.eqb (length answers) (length params));
    [ | reflexivity ].
  assert (Hcanonicals :
    map snd (map (rename_answer r) answers)
    = map (rename_simple r) (map snd answers)).
  { rewrite !map_map. reflexivity. }
  rewrite Hcanonicals, bindings_renamed.
  apply bind_params_to_args_equivariant.
  intros s. rewrite simple_in_terms_b_renamed, not_among_renamed.
  reflexivity.
Qed.

Theorem leaf_uses_equivariant : leaf_uses_equivariant_statement.
Proof.
  intros r tenv k e.
  destruct e
    as [x declared d body | bindings body | k' h body | handlers body | ap
        | ac | scrutinee arms | ];
    cbn [rename_expr leaf_uses option_map]; try reflexivity.
  - cbn [rename_apply_cont ac_continuation ac_args].
    rewrite rename_continuation_eqb.
    destruct (Pos.eqb (ac_continuation ac) k); reflexivity.
  - f_equal.
    induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
    cbn [map flat_map fst snd rename_apply_cont ac_continuation ac_args].
    rewrite rename_continuation_eqb, map_app, <- IH.
    destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
    cbn [map app rename_application fst snd].
    rewrite arm_env_equivariant. reflexivity.
Qed.

Theorem let_env_equivariant : let_env_equivariant_statement.
Proof.
  intros r tenv x declared d. unfold let_env.
  rewrite resolve_value_slot_equivariant, simplify_let_equivariant.
  destruct (simplify_let tenv x (resolve_value_slot tenv declared d))
    as [ | d' tenv' | ];
    cbn [rename_let_step option_map]; try reflexivity.
  apply declare_variable_equivariant.
Qed.

Theorem find_use_equivariant : find_use_equivariant_statement.
Proof.
  intros r scope tenv k h body. revert tenv.
  induction body
    as [x declared d rest IH | bindings rest _ | k' h' rest IH
        | handlers rest _ | ap | ac | scrutinee arms | ];
    intros tenv; cbn [rename_expr find_use option_map]; try reflexivity.
  - rewrite var_free_in_handler_b_equivariant.
    destruct (var_free_in_handler_b x h); [ reflexivity | ].
    rewrite let_env_equivariant.
    destruct (let_env tenv x declared d) as [tenv' | ]; cbn [option_map];
      [ | reflexivity ].
    rewrite IH.
    destruct (find_use scope tenv' k h rest) as [[[target xs] ks] | ];
      reflexivity.
  - rewrite rename_continuation_eqb, !cont_free_in_handler_b_equivariant.
    destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k h')
              && negb (cont_free_in_handler_b k' h));
      [ | reflexivity ].
    rewrite IH.
    destruct (find_use scope tenv k h rest) as [[[target xs] ks] | ];
      reflexivity.
  - destruct h as [params metadata handler]. cbn [rename_handler].
    change (Apply_cont (rename_apply_cont r ac))
      with (rename_expr r (Apply_cont ac)).
    rewrite leaf_uses_equivariant.
    destruct (leaf_uses tenv k (Apply_cont ac))
      as [[ | [tenv_use args] [ | second others]] | ];
      cbn [option_map map rename_application fst snd]; try reflexivity.
    rewrite use_env_equivariant.
    destruct (use_env scope tenv_use params metadata args) as [target | ];
      reflexivity.
  - destruct h as [params metadata handler]. cbn [rename_handler].
    change (Switch (rename_simple r scrutinee)
              (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                 arms))
      with (rename_expr r (Switch scrutinee arms)).
    rewrite leaf_uses_equivariant.
    destruct (leaf_uses tenv k (Switch scrutinee arms))
      as [[ | [tenv_use args] [ | second others]] | ];
      cbn [option_map map rename_application fst snd]; try reflexivity.
    rewrite use_env_equivariant.
    destruct (use_env scope tenv_use params metadata args) as [target | ];
      reflexivity.
Qed.

Theorem binders_not_free_equivariant
    : binders_not_free_equivariant_statement.
Proof.
  intros r xs ks h. unfold binders_not_free. f_equal.
  - induction xs as [ | x xs IH]; [ reflexivity | ].
    cbn [map forallb]. rewrite var_free_in_handler_b_equivariant, IH.
    reflexivity.
  - induction ks as [ | k ks IH]; [ reflexivity | ].
    cbn [map forallb]. rewrite cont_free_in_handler_b_equivariant, IH.
    reflexivity.
Qed.

Fixpoint gather_uses_equivariant_aux (r : renaming) (body : expr)
    {struct body} :
  ∀ (tenv : Typing_env.t) (k : continuation) (h : continuation_handler),
    gather_uses (rename_typing_env r tenv) (rename_continuation r k)
      (rename_handler r h) (rename_expr r body)
    = option_map (rename_gathered r) (gather_uses tenv k h body).
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros tenv k h; cbn [rename_expr gather_uses option_map];
    try reflexivity.
  - rewrite var_free_in_handler_b_equivariant.
    destruct (var_free_in_handler_b x h); [ reflexivity | ].
    rewrite let_env_equivariant.
    destruct (let_env tenv x declared d) as [tenv' | ]; cbn [option_map];
      [ | reflexivity ].
    rewrite (gather_uses_equivariant_aux r rest tenv' k h).
    destruct (gather_uses tenv' k h rest) as [[[uses xs] ks] | ];
      reflexivity.
  - rewrite rename_continuation_eqb, !cont_free_in_handler_b_equivariant.
    destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k' h));
      [ | reflexivity ].
    rewrite (gather_uses_equivariant_aux r rest tenv k h).
    destruct (gather_uses tenv k h rest) as [[[uses xs] ks] | ];
      cbn [option_map rename_gathered fst snd]; [ | reflexivity ].
    destruct (cont_free_in_handler_b k h'); [ | reflexivity ].
    destruct h' as [params' metadata' body']. cbn [rename_handler].
    assert (Hparams :
      forallb
        (fun x => negb (var_free_in_handler_b x (rename_handler r h)))
        (map (rename_variable r) params')
      = forallb (fun x => negb (var_free_in_handler_b x h)) params').
    { induction params' as [ | x params' IH]; [ reflexivity | ].
      cbn [map forallb]. rewrite var_free_in_handler_b_equivariant, IH.
      reflexivity. }
    rewrite Hparams.
    destruct (negb (cont_free_in_handler_b k h)
              && forallb (fun x => negb (var_free_in_handler_b x h)) params');
      [ | reflexivity ].
    rewrite handler_env_rename.
    destruct (handler_env tenv params' metadata') as [tenv_handler | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite (gather_uses_equivariant_aux r body' tenv_handler k h).
    destruct (gather_uses tenv_handler k h body') as [[[uses' xs'] ks'] | ];
      cbn [option_map rename_gathered fst snd]; [ | reflexivity ].
    unfold rename_gathered. cbn [fst snd map]. rewrite !map_app. reflexivity.
  - change (Apply_cont (rename_apply_cont r ac))
      with (rename_expr r (Apply_cont ac)).
    rewrite leaf_uses_equivariant.
    destruct (leaf_uses tenv k (Apply_cont ac)); reflexivity.
  - change (Switch (rename_simple r scrutinee)
              (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                 arms))
      with (rename_expr r (Switch scrutinee arms)).
    rewrite leaf_uses_equivariant.
    destruct (leaf_uses tenv k (Switch scrutinee arms)); reflexivity.
Qed.

Theorem gather_uses_equivariant : gather_uses_equivariant_statement.
Proof.
  intros r tenv k h body. apply gather_uses_equivariant_aux.
Qed.

Lemma find_gathered_use_equivariant (r : renaming) (tenv : Typing_env.t)
    (k : continuation) (h : continuation_handler) (body : expr) :
  find_gathered_use (rename_typing_env r tenv) (rename_continuation r k)
    (rename_handler r h) (rename_expr r body)
  = option_map (rename_use r) (find_gathered_use tenv k h body).
Proof.
  destruct h as [params metadata handler].
  unfold find_gathered_use. cbn [rename_handler].
  change (Handler (map (rename_variable r) params) metadata
            (rename_expr r handler))
    with (rename_handler r (Handler params metadata handler)).
  rewrite gather_uses_equivariant.
  destruct (gather_uses tenv k (Handler params metadata handler) body)
    as [[[uses xs] ks] | ];
    cbn [option_map rename_gathered fst snd]; [ | reflexivity ].
  destruct uses as [ | [tenv_use args] [ | second others]];
    cbn [map rename_application fst snd]; try reflexivity.
  rewrite use_env_equivariant.
  destruct (use_env tenv tenv_use params metadata args) as [target | ];
    reflexivity.
Qed.

Theorem find_one_use_equivariant : find_one_use_equivariant_statement.
Proof.
  intros r tenv k h body. unfold find_one_use.
  rewrite find_use_equivariant, find_gathered_use_equivariant.
  destruct (find_use tenv tenv k h body) as [found | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

Fixpoint simplify_expr_equivariant_aux (r : renaming) (e : expr) {struct e} :
  ∀ tenv : Typing_env.t,
    simplify_expr (rename_typing_env r tenv) (rename_expr r e)
    = rename_expr r (simplify_expr tenv e)
with simplify_handler_equivariant (r : renaming) (h : continuation_handler)
    {struct h} :
  ∀ (recursive : bool) (tenv : Typing_env.t),
    simplify_handler recursive (rename_typing_env r tenv)
      (rename_handler r h)
    = rename_handler r (simplify_handler recursive tenv h)
with simplify_static_binding_equivariant (r : renaming)
    (binding : static_binding) {struct binding} :
  ∀ tenv : Typing_env.t,
    simplify_static_binding (rename_typing_env r tenv)
      (rename_static_binding r binding)
    = rename_static_binding r (simplify_static_binding tenv binding)
with simplify_code_equivariant (r : renaming) (c : code) {struct c} :
  ∀ tenv : Typing_env.t,
    simplify_code (rename_typing_env r tenv) (rename_code r c)
    = rename_code r (simplify_code tenv c).
Proof.
  - destruct e
      as [x declared d body | bindings body | k handler body
          | handlers body | ap | ac | scrutinee arms | ];
      intros tenv; cbn [rename_expr simplify_expr].
    + rewrite resolve_value_slot_equivariant, simplify_let_equivariant.
      destruct (simplify_let tenv x (resolve_value_slot tenv declared d))
        as [ | d' tenv' | ];
        cbn [rename_let_step rename_expr]; try reflexivity.
      * rewrite (simplify_expr_equivariant_aux r body tenv'). reflexivity.
      * rewrite declare_variable_equivariant.
        destruct (declare_variable tenv x declared) as [tenv' | ];
          cbn [option_map rename_expr]; [ | reflexivity ].
        rewrite (simplify_expr_equivariant_aux r body tenv'). reflexivity.
    + cbv zeta. rewrite add_static_bindings_rename.
      rewrite (simplify_expr_equivariant_aux r body). f_equal.
      generalize (add_static_bindings tenv bindings). intros tenv'.
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ reflexivity | ].
      rewrite (simplify_static_binding_equivariant r binding tenv'), IH.
      reflexivity.
    + pose proof (simplify_handler_equivariant r handler false tenv)
        as Hdefault.
      rewrite (simplify_expr_equivariant_aux r body tenv).
      destruct handler as [params metadata handler].
      cbn [rename_handler].
      change (Handler (map (rename_variable r) params) metadata
                (rename_expr r handler))
        with (rename_handler r (Handler params metadata handler)).
      rewrite find_one_use_equivariant.
      destruct (find_one_use tenv k (Handler params metadata handler)
                  (simplify_expr tenv body))
        as [[[target xs] ks] | ];
        cbn [option_map rename_use fst snd];
        [ | rewrite Hdefault; reflexivity ].
      rewrite (simplify_expr_equivariant_aux r handler target).
      change (Handler (map (rename_variable r) params) metadata
                (rename_expr r (simplify_expr target handler)))
        with (rename_handler r
                (Handler params metadata (simplify_expr target handler))).
      rewrite binders_not_free_equivariant.
      destruct (binders_not_free xs ks
                  (Handler params metadata (simplify_expr target handler)));
        [ reflexivity | ].
      rewrite Hdefault. reflexivity.
    + rewrite (simplify_expr_equivariant_aux r body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map];
        [ reflexivity | ].
      rewrite (simplify_handler_equivariant r h true tenv), IH.
      reflexivity.
    + apply simplify_apply_rename.
    + rewrite simplify_apply_cont_equivariant. reflexivity.
    + apply simplify_switch_equivariant.
    + reflexivity.
  - destruct h as [params metadata handler]; intros recursive tenv;
      cbn [rename_handler simplify_handler].
    rewrite handler_env_rename.
    destruct (handler_env tenv params metadata) as [tenv_handler | ];
      cbn [option_map]; [ | reflexivity ].
    cbv zeta.
    rewrite (simplify_expr_equivariant_aux r handler tenv_handler),
      !rename_binds_statics.
    destruct (negb recursive
              || Bool.eqb (binds_statics handler)
                   (binds_statics (simplify_expr tenv_handler handler)));
      reflexivity.
  - destruct binding as [id c | id | sym const]; intros tenv;
      cbn [rename_static_binding simplify_static_binding];
      [ rewrite (simplify_code_equivariant r c tenv) | | ]; reflexivity.
  - destruct c as [params my_closure ret exn metadata body]; intros tenv;
      cbn [rename_code simplify_code].
    rewrite code_env_in_rename.
    destruct (code_env_in tenv params my_closure metadata)
      as [tenv_code | ];
      cbn [option_map]; [ | reflexivity ].
    cbv zeta.
    rewrite (simplify_expr_equivariant_aux r body tenv_code),
      !rename_binds_statics.
    destruct (negb (binds_statics body)
              && negb (binds_statics (simplify_expr tenv_code body)));
      reflexivity.
Qed.

Theorem simplify_expr_equivariant : simplify_expr_equivariant_statement.
Proof. intros r tenv e. apply simplify_expr_equivariant_aux. Qed.

Theorem simplify_by_types_equivariant
    : simplify_by_types_equivariant_statement.
Proof.
  intros r e. unfold simplify_by_types.
  rewrite <- (create_equivariant r) at 1. apply simplify_expr_equivariant.
Qed.

Theorem rewrite_handler_at_use_equivariant
    : rewrite_handler_at_use_equivariant_statement.
Proof.
  intros r tenv k [params metadata handler] body.
  unfold rewrite_handler_at_use. cbn [rename_handler].
  change (Handler (map (rename_variable r) params) metadata
            (rename_expr r handler))
    with (rename_handler r (Handler params metadata handler)).
  rewrite find_use_equivariant.
  destruct (find_use tenv tenv k (Handler params metadata handler) body)
    as [[[target xs] ks] | ];
    cbn [option_map rename_use fst snd]; [ | reflexivity ].
  cbv zeta. rewrite simplify_expr_equivariant.
  change (Handler (map (rename_variable r) params) metadata
            (rename_expr r (simplify_expr target handler)))
    with (rename_handler r
            (Handler params metadata (simplify_expr target handler))).
  rewrite binders_not_free_equivariant.
  destruct (binders_not_free xs ks
              (Handler params metadata (simplify_expr target handler)));
    reflexivity.
Qed.
