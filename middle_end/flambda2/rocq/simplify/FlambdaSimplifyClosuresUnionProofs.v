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



(* Proofs of the statements of FlambdaSimplifyClosuresUnionSpec.v. This
   file needs no review: each statement is closed by a [Theorem] whose
   type is the statement. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNormaliseSpec
  FlambdaUnusedContsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaTypeEqualSpec
  FlambdaTypeEqualProofs FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaTypingEnvRefinementSpec FlambdaTypingEnvRefinementProofs
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaExpandHeadSpec FlambdaExpandHeadProofs FlambdaProvers
  FlambdaProversRenameSpec FlambdaProversRenameProofs
  FlambdaEnvRenameJoinTypesProofs FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyPrimitivesProofs FlambdaSimplifyPrimitivesRenamingSpec
  FlambdaSimplifyPrimitivesRenamingProofs FlambdaSimplifyApplySpec
  FlambdaSimplifyCodeEnvSpec FlambdaSimplifyHandlersSpec
  FlambdaSimplifyHandlersProofs FlambdaSimplifyPassSpec
  FlambdaSimplifyPassProofs FlambdaSimplifyUsesSpec FlambdaSimplifyUsesProofs
  FlambdaSimplifyPassRenamingSpec FlambdaSimplifyPassRenamingProofs
  FlambdaSimplifyPruningSpec FlambdaSimplifyPruningProofs
  FlambdaSimplifyClosuresUnionSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* A type that is known to be of closures                                   *)
(* ======================================================================== *)

(* What it is to have the type of some closures: to be the closure at
   a function slot that has a case, which admits it. *)
Lemma closures_type_satisfied (statics : static_env) (vars : env) (v : value)
    (closures : row_like_for_closures) :
  type_satisfied statics vars v (closures_type closures)
  <-> ∃ (slot : function_slot) (functions : list (function_slot * code_id))
        (value_slots : list (value_slot * value))
        (case : row_like_closures_case),
        closure_in_set statics v = Some (slot, functions, value_slots)
        ∧ lookup slot (cases_of closures) = Some case
        ∧ row_like_closures_case_satisfied statics vars v slot functions
            value_slots case.
Proof.
  unfold closures_type, create_closures.
  rewrite non_null_value_satisfied, closures_satisfied.
  destruct closures as [known_closures]. cbn [cases_of]. split.
  - intros [slot [functions [value_slots [Hclosure Hrow]]]].
    apply row_like_for_closures_lookup in Hrow.
    destruct Hrow as [case [Hlookup Hcase]].
    exists slot, functions, value_slots, case.
    split; [ exact Hclosure | split; assumption ].
  - intros [slot [functions [value_slots [case [Hclosure [Hlookup Hcase]]]]]].
    exists slot, functions, value_slots. split; [ exact Hclosure | ].
    apply row_like_for_closures_lookup. exists case.
    split; assumption.
Qed.

Theorem closures_known_sound : closures_known_sound_statement.
Proof.
  intros statics valuation tenv ty closures v Hsat Hty Hknown.
  unfold closures_known in Hknown.
  destruct (expand_head tenv ty) as [expanded | ] eqn:Hexpand;
    [ | discriminate ].
  pose proof (expand_head_sound statics valuation tenv ty expanded v Hsat Hty
                Hexpand)
    as Hexpanded.
  unfold Expanded_type.to_type in Hexpanded.
  destruct (Expanded_type.descr expanded)
    as [ | [head | head | k head | head | head] | ];
    try discriminate.
  unfold closures_of_head in Hknown.
  destruct head
    as [[ | [is_int immediates get_tag blocks extensions is_unique | alloc
            | k contents alloc | closures' alloc | strs]
         | ]
        [ | is_null]];
    try discriminate.
  injection Hknown as <-.
  change
    (type_satisfied statics valuation v
       (non_null_value (Closures closures' alloc)))
    in Hexpanded.
  apply non_null_value_satisfied in Hexpanded.
  unfold closures_type, create_closures. apply non_null_value_satisfied.
  exact Hexpanded.
Qed.

(* ======================================================================== *)
(* The union                                                                *)
(* ======================================================================== *)

(* What it is to have the type of the closures of one case. *)
Lemma one_case_type_satisfied (statics : static_env) (vars : env) (v : value)
    (slot : function_slot) (case : row_like_closures_case) :
  type_satisfied statics vars v (one_case_type slot case)
  <-> ∃ (functions : list (function_slot * code_id))
        (value_slots : list (value_slot * value)),
        closure_in_set statics v = Some (slot, functions, value_slots)
        ∧ row_like_closures_case_satisfied statics vars v slot functions
            value_slots case.
Proof.
  unfold one_case_type. rewrite closures_type_satisfied. cbn [cases_of].
  split.
  - intros [slot' [functions [value_slots [case' [Hclosure [Hlookup Hcase]]]]]].
    simpl in Hlookup.
    destruct (Pos.eqb_spec slot' slot) as [-> | _]; [ | discriminate ].
    injection Hlookup as <-. exists functions, value_slots.
    split; assumption.
  - intros [functions [value_slots [Hclosure Hcase]]].
    exists slot, functions, value_slots, case.
    split; [ exact Hclosure | split; [ | exact Hcase ] ].
    simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* The test, read back: a slot that both have has the same case. *)
Lemma closures_agree_lookup (closures closures' : row_like_for_closures)
    (slot : function_slot) (case case' : row_like_closures_case) :
  closures_agree closures closures' = true ->
  lookup slot (cases_of closures) = Some case ->
  lookup slot (cases_of closures') = Some case' ->
  type_eqb (one_case_type slot case) (one_case_type slot case') = true.
Proof.
  unfold closures_agree. intros Hagree Hcase Hcase'.
  rewrite forallb_forall in Hagree.
  assert (Hin : slot ∈ domain (cases_of closures)).
  { apply among_In. rewrite among_domain. unfold is_bound. rewrite Hcase.
    reflexivity. }
  specialize (Hagree slot Hin). rewrite Hcase, Hcase' in Hagree.
  exact Hagree.
Qed.

Theorem closures_union_satisfied : closures_union_satisfied_statement.
Proof.
  intros statics vars closures closures' union v Hunion.
  unfold closures_union in Hunion.
  destruct (closures_agree closures closures') eqn:Hagree; [ | discriminate ].
  injection Hunion as <-. rewrite !closures_type_satisfied. cbn [cases_of].
  split.
  - intros [slot [functions [value_slots [case [Hclosure [Hlookup Hcase]]]]]].
    rewrite lookup_union_with in Hlookup.
    destruct (lookup slot (cases_of closures)) as [case0 | ] eqn:Hfirst.
    + left. exists slot, functions, value_slots, case.
      split; [ exact Hclosure | split; [ | exact Hcase ] ].
      rewrite Hfirst. destruct (lookup slot (cases_of closures'));
        exact Hlookup.
    + right. exists slot, functions, value_slots, case.
      split; [ exact Hclosure | split; assumption ].
  - intros [Hhas | Hhas];
      destruct Hhas
        as [slot [functions [value_slots [case [Hclosure [Hlookup Hcase]]]]]].
    + exists slot, functions, value_slots, case.
      split; [ exact Hclosure | split; [ | exact Hcase ] ].
      rewrite lookup_union_with, Hlookup.
      destruct (lookup slot (cases_of closures')); reflexivity.
    + destruct (lookup slot (cases_of closures)) as [case0 | ] eqn:Hfirst.
      * (* Both have a case for the slot: the two are the same. *)
        exists slot, functions, value_slots, case0.
        split; [ exact Hclosure | split ].
        -- rewrite lookup_union_with, Hfirst, Hlookup. reflexivity.
        -- pose proof (closures_agree_lookup closures closures' slot case0
                         case Hagree Hfirst Hlookup)
             as Hsame.
           pose proof (proj2
                         (type_equal_satisfied statics vars v _ _ Hsame))
             as Hboth.
           rewrite !one_case_type_satisfied in Hboth.
           destruct (Hboth
                       (ex_intro _ functions
                          (ex_intro _ value_slots (conj Hclosure Hcase))))
             as [functions' [value_slots' [Hclosure' Hcase']]].
           rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
           exact Hcase'.
      * exists slot, functions, value_slots, case.
        split; [ exact Hclosure | split; [ | exact Hcase ] ].
        rewrite lookup_union_with, Hfirst. exact Hlookup.
Qed.

(* The union of a list: a value has its type exactly when it has the
   type of one of the list. *)
Lemma closures_union_all_satisfied (statics : static_env) (vars : env)
    (v : value) (rest : list row_like_for_closures) :
  ∀ first union : row_like_for_closures,
    closures_union_all first rest = Some union ->
    (type_satisfied statics vars v (closures_type union)
     <-> ∃ closures : row_like_for_closures,
           closures ∈ first :: rest
           ∧ type_satisfied statics vars v (closures_type closures)).
Proof.
  induction rest as [ | closures rest IH]; intros first union;
    cbn [closures_union_all].
  - intros [= <-]. split.
    + intros Hfirst. exists first. split; [ left; reflexivity | exact Hfirst ].
    + intros [closures [[<- | []] Hhas]]. exact Hhas.
  - destruct (closures_union first closures) as [both | ] eqn:Hboth;
      [ | discriminate ].
    intros Hrest. rewrite (IH both union Hrest). split.
    + intros [closures0 [[<- | Hin] Hhas]].
      * apply (closures_union_satisfied statics vars first closures both v
                 Hboth)
          in Hhas.
        destruct Hhas as [Hhas | Hhas].
        -- exists first. split; [ left; reflexivity | exact Hhas ].
        -- exists closures.
           split; [ right; left; reflexivity | exact Hhas ].
      * exists closures0. split; [ right; right; exact Hin | exact Hhas ].
    + intros [closures0 [[<- | [<- | Hin]] Hhas]].
      * exists both. split; [ left; reflexivity | ].
        apply (closures_union_satisfied statics vars first closures both v
                 Hboth).
        left. exact Hhas.
      * exists both. split; [ left; reflexivity | ].
        apply (closures_union_satisfied statics vars first closures both v
                 Hboth).
        right. exact Hhas.
      * exists closures0. split; [ right; exact Hin | exact Hhas ].
Qed.

(* ======================================================================== *)
(* The types of the parameters                                              *)
(* ======================================================================== *)

(* What [map_opt] answers has the image of each element. *)
Lemma map_opt_in {A B : Type} (f : A -> option B) (l : list A) :
  ∀ (l' : list B) (a : A),
    map_opt f l = Some l' -> a ∈ l -> ∃ b : B, f a = Some b ∧ b ∈ l'.
Proof.
  induction l as [ | a0 l IH]; intros l' a; simpl.
  - intros _ [].
  - destruct (f a0) as [b0 | ] eqn:Hb0; [ | discriminate ].
    destruct (map_opt f l) as [l0 | ] eqn:Hl0; [ | discriminate ].
    simpl. intros [= <-] [<- | Hin].
    + exists b0. split; [ exact Hb0 | left; reflexivity ].
    + destruct (IH l0 a eq_refl Hin) as [b [Hb Hin']]. exists b.
      split; [ exact Hb | right; exact Hin' ].
Qed.

(* Under a valuation that satisfies the environment of one of the
   applications and agrees with the machine: the value of the argument
   at the position has the type of the union. *)
Lemma union_of_arguments_satisfied (uses : list (Typing_env.t * list simple))
    (index : nat) (ty : flambda_type) (use : Typing_env.t * list simple)
    (statics : static_env) (valuation vars : env) (arg : simple)
    (v : value) :
  union_of_arguments uses index = Some ty ->
  use ∈ uses ->
  typing_env_satisfied_strictly statics valuation (fst use) ->
  agrees_in_terms (fst use) valuation vars ->
  nth_error (snd use) index = Some arg ->
  eval_simple vars arg = Some v ->
  type_satisfied statics valuation v ty.
Proof.
  unfold union_of_arguments. intros Hunion Hin Hsat Hagrees Hnth Heval.
  destruct (map_opt (argument_closures index) uses) as [[ | first rest] | ]
    eqn:Hall; try discriminate.
  destruct (closures_union_all first rest) as [union | ] eqn:Hrest;
    [ | discriminate ].
  injection Hunion as <-.
  destruct (map_opt_in _ uses _ use Hall Hin) as [closures [Hclosures Hamong]].
  apply (closures_union_all_satisfied statics valuation v rest first union
           Hrest).
  exists closures. split; [ exact Hamong | ].
  unfold argument_closures in Hclosures. rewrite Hnth in Hclosures.
  destruct (simplify_simple (fst use) arg) as [[ty_arg canonical] | ]
    eqn:Hsimple; [ | discriminate ].
  apply (closures_known_sound statics valuation (fst use) ty_arg closures v
           Hsat);
    [ | exact Hclosures ].
  unfold simplify_simple in Hsimple.
  destruct (Typing_env.mem_simple (Some Name_mode.normal) (fst use) arg)
    eqn:Hmem; [ | discriminate ].
  destruct (Typing_env.type_simple_in_term_exn (fst use)
              (Some Name_mode.normal) arg)
    as [[answer | ] | ] eqn:Htype; try discriminate.
  injection Hsimple as ->.
  apply (proj2 (type_simple_in_term_satisfied (fun _ => False) statics
                  valuation (fst use) (Some Name_mode.normal) arg canonical
                  ty_arg Hsat Htype)).
  rewrite (agrees_eval_simple (fst use) valuation vars arg Hagrees);
    [ exact Heval | ].
  intros x ->. exact Hmem.
Qed.

(* In the machine, at one of the applications: the value of the
   argument at a position has the type that the parameter there is
   given, under the variables of the machine; and the type speaks only
   of variables of terms of the [Let_cont]. *)
Lemma parameter_union_in_machine (tenv : Typing_env.t)
    (uses : list (Typing_env.t * list simple)) (index : nat)
    (kind : flambda_kind) (ty : flambda_type)
    (use : Typing_env.t * list simple) (statics : static_env) (vars : env)
    (vs : list value) (v : value) :
  parameter_union tenv uses index kind = Some ty ->
  use ∈ uses ->
  (∀ y : variable, in_scope_in_terms tenv y -> in_scope_in_terms (fst use) y)
  ->
  env_satisfies_strictly (fst use) statics vars ->
  eval_simples vars (snd use) = Some vs ->
  nth_error vs index = Some v ->
  (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y)
  ∧ type_satisfied statics vars v ty.
Proof.
  unfold parameter_union.
  intros Hunion Hin Hscope [Hbound [valuation [Hsat Hagrees]]] Hargs Hnth.
  destruct kind; try discriminate.
  destruct (union_of_arguments uses index) as [ty' | ] eqn:Hty;
    [ | discriminate ].
  destruct (type_in_scope_b tenv ty') eqn:Hin_scope; [ | discriminate ].
  injection Hunion as <-.
  assert (Hfree :
    ∀ y : variable, y ∈ free_variables ty' -> in_scope_in_terms tenv y).
  { intros y Hy. unfold type_in_scope_b in Hin_scope.
    rewrite forallb_forall in Hin_scope. exact (Hin_scope y Hy). }
  split; [ exact Hfree | ].
  destruct (forall2_nth_right _ _ _ index v
              (eval_simples_each vars (snd use) vs Hargs) Hnth)
    as [arg [Hnth_arg Heval]].
  apply (proj1 (type_satisfied_coincide statics valuation vars v ty'
                  (fun y Hy => Hagrees y (Hscope y (Hfree y Hy))))).
  exact (union_of_arguments_satisfied uses index ty' use statics valuation
           vars arg v Hty Hin Hsat Hagrees Hnth_arg Heval).
Qed.

(* ======================================================================== *)
(* The environment                                                          *)
(* ======================================================================== *)

(* What is known of the value that a variable defined is bound to: it
   is of the kind, and it has the type, if there is one, under the
   variables of the machine; the type speaks only of variables of
   terms. *)
Definition definition_holds (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (definition : variable * flambda_kind * option flambda_type)
    (v : value) : Prop :=
  kind_of_value v = snd (fst definition)
  ∧ ∀ ty : flambda_type,
      snd definition = Some ty ->
      (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y)
      ∧ type_satisfied statics vars v ty.

(* The environment in which variables have been defined, some with
   types, is satisfied once the machine has bound them to values of
   which that is known. *)
Lemma define_typed_checked_satisfied
    (definitions : list (variable * flambda_kind * option flambda_type)) :
  ∀ (tenv tenv' : Typing_env.t) (statics : static_env) (vars vars' : env)
    (vs : list value),
    define_typed_checked tenv definitions = Some tenv' ->
    Forall2 (definition_holds tenv statics vars) definitions vs ->
    bind_params (map (fun definition => fst (fst definition)) definitions) vs
      vars
    = Some vars' ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly tenv' statics vars'.
Proof.
  induction definitions as [ | [[x kind] ty] definitions IH];
    intros tenv tenv' statics vars vars' vs Hdefined Hholds Hbind Hsat.
  - inversion Hholds; subst. simpl in Hdefined, Hbind.
    injection Hdefined as <-. injection Hbind as <-. exact Hsat.
  - inversion Hholds as [ | definition v definitions' vs' [Hkind Hty] Hrest];
      subst.
    cbn [define_typed_checked] in Hdefined. simpl in Hbind.
    cbn [fst snd] in Hkind, Hty.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Hwf Hfresh].
    apply negb_true_iff in Hfresh. cbv zeta in Hdefined.
    set (defined :=
           Typing_env.add_variable_definition tenv x kind Name_mode.normal)
      in *.
    assert (Hdefined_sat :
      env_satisfies_strictly defined statics ((x, v) :: vars)).
    { apply add_variable_definition_env_satisfies_strictly;
        [ exact Hwf | exact Hfresh | exact Hkind | exact Hsat ]. }
    assert (Hscope :
      ∀ y : variable,
        in_scope_in_terms tenv y -> in_scope_in_terms defined y ∧ y <> x).
    { intros y Hy.
      pose proof (not_bound_not_in_scope tenv x y Hfresh Hy) as Hne.
      split; [ | exact Hne ].
      apply (add_variable_definition_in_scope tenv x kind y Hwf). right.
      split; assumption. }
    set (tenv1 :=
           match ty with
           | Some ty => Typing_env.replace_equation defined (Name_var x) ty
           | None => defined
           end)
      in *.
    assert (Hscope1 :
      ∀ y : variable, in_scope_in_terms tenv y -> in_scope_in_terms tenv1 y).
    { intros y Hy. unfold tenv1.
      destruct ty as [ty | ]; [ apply replace_equation_in_scope | ];
        exact (proj1 (Hscope y Hy)). }
    assert (Hsat1 : env_satisfies_strictly tenv1 statics ((x, v) :: vars)).
    { unfold tenv1. destruct ty as [ty | ]; [ | exact Hdefined_sat ].
      destruct (Hty ty eq_refl) as [Hfree Hhas].
      apply replace_equation_env_satisfies_strictly;
        [ | | | exact Hdefined_sat ].
      - apply (add_variable_definition_in_scope tenv x kind x Hwf). left.
        reflexivity.
      - intros y Hy. exact (proj1 (Hscope y (Hfree y Hy))).
      - intros v' Hlookup.
        pose proof (eq_trans (eq_sym Hlookup) (lookup_head x v vars)) as Heq.
        injection Heq as ->.
        apply (proj1 (type_satisfied_coincide statics vars ((x, v) :: vars)
                        v ty
                        (fun y Hy =>
                           eq_sym
                             (lookup_other y x v vars
                                (proj2 (Hscope y (Hfree y Hy))))))).
        exact Hhas. }
    apply (IH tenv1 tenv' statics ((x, v) :: vars) vars' vs' Hdefined);
      [ | exact Hbind | exact Hsat1 ].
    eapply Forall2_impl; [ | exact Hrest ].
    intros definition' v' [Hkind' Hty']. split; [ exact Hkind' | ].
    intros ty' Heq. destruct (Hty' ty' Heq) as [Hfree' Hhas']. split.
    + intros y Hy. exact (Hscope1 y (Hfree' y Hy)).
    + apply (proj1 (type_satisfied_coincide statics vars ((x, v) :: vars) v'
                      ty'
                      (fun y Hy =>
                         eq_sym
                           (lookup_other y x v vars
                              (proj2 (Hscope y (Hfree' y Hy))))))).
      exact Hhas'.
Qed.

(* There are as many types as kinds. *)
Lemma parameter_unions_length (scope : Typing_env.t)
    (uses : list (Typing_env.t * list simple)) (kinds : list flambda_kind) :
  ∀ index : nat,
    length (parameter_unions scope uses index kinds) = length kinds.
Proof.
  induction kinds as [ | kind kinds IH]; intros index; simpl;
    [ reflexivity | ].
  f_equal. apply IH.
Qed.

(* At one of the applications, what is known of the values of the
   arguments from a position on. *)
Lemma parameter_unions_hold (tenv : Typing_env.t)
    (uses : list (Typing_env.t * list simple))
    (use : Typing_env.t * list simple) (statics : static_env) (vars : env)
    (all_vs : list value) :
  use ∈ uses ->
  (∀ y : variable, in_scope_in_terms tenv y -> in_scope_in_terms (fst use) y)
  ->
  env_satisfies_strictly (fst use) statics vars ->
  eval_simples vars (snd use) = Some all_vs ->
  ∀ (kinds : list flambda_kind) (params : list variable) (index : nat)
    (vs : list value),
    length kinds = length params ->
    map kind_of_value vs = kinds ->
    (∀ (i : nat) (v : value),
      nth_error vs i = Some v -> nth_error all_vs (index + i)%nat = Some v) ->
    Forall2 (definition_holds tenv statics vars)
      (combine (combine params kinds)
         (parameter_unions tenv uses index kinds))
      vs.
Proof.
  intros Hin Hscope Hsat Hargs.
  induction kinds as [ | kind kinds IH];
    intros [ | param params] index [ | v vs] Hlength Hkinds Hnth;
    simpl in Hlength, Hkinds; try discriminate.
  - apply Forall2_nil.
  - injection Hkinds as Hkind Hkinds. injection Hlength as Hlength.
    cbn [parameter_unions combine]. apply Forall2_cons.
    + unfold definition_holds. cbn [fst snd]. split; [ exact Hkind | ].
      intros ty Hty.
      apply (parameter_union_in_machine tenv uses index kind ty use statics
               vars all_vs v Hty Hin Hscope Hsat Hargs).
      rewrite <- (Nat.add_0_r index). exact (Hnth 0%nat v eq_refl).
    + apply (IH params (S index) vs Hlength Hkinds). intros i v' Hi.
      replace (S index + i)%nat with (index + S i)%nat by lia.
      exact (Hnth (S i) v' Hi).
Qed.

Theorem closures_union_env_satisfied : closures_union_env_satisfied_statement.
Proof.
  intros tenv params metadata uses target use statics vars vs handler_vars
    Henv Hin Hscope Hsat Hsat_use Hargs Hkinds Hbind.
  unfold closures_union_env in Henv. cbv zeta in Henv.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params))
    eqn:Hlength; [ | discriminate ].
  apply Nat.eqb_eq in Hlength.
  apply (define_typed_checked_satisfied _ tenv target statics vars
           handler_vars vs Henv);
    [ | | exact Hsat ].
  - apply (parameter_unions_hold tenv uses use statics vars vs Hin Hscope
             Hsat_use Hargs);
      [ exact Hlength | exact Hkinds | ].
    intros i v Hi. exact Hi.
  - rewrite <- (map_map fst fst), map_fst_combine, map_fst_combine;
      [ exact Hbind | exact Hlength | ].
    rewrite parameter_unions_length, length_combine, Hlength.
    symmetry. apply Nat.min_id.
Qed.

(* With no application no parameter is given a type. *)
Lemma define_typed_no_use (scope : Typing_env.t) (kinds : list flambda_kind) :
  ∀ (params : list variable) (index : nat) (tenv : Typing_env.t),
    define_typed_checked tenv
      (combine (combine params kinds) (parameter_unions scope [] index kinds))
    = define_variables_checked tenv (combine params kinds).
Proof.
  induction kinds as [ | kind kinds IH]; intros [ | param params] index tenv;
    try reflexivity.
  cbn [parameter_unions combine define_typed_checked define_variables_checked].
  assert (Hnone : parameter_union scope [] index kind = None)
    by (destruct kind; reflexivity).
  rewrite Hnone.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var param)));
    [ | reflexivity ].
  apply IH.
Qed.

Theorem closures_union_env_no_use : closures_union_env_no_use_statement.
Proof.
  intros tenv params metadata. unfold closures_union_env, handler_env.
  cbv zeta.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params));
    [ | reflexivity ].
  apply define_typed_no_use.
Qed.

(* ======================================================================== *)
(* The handler rewritten                                                    *)
(* ======================================================================== *)

Theorem let_cont_handler_closures_union
  : let_cont_handler_closures_union_statement.
Proof.
  intros tenv k params metadata handler handler' body uses target Hleaf Henv
    Hhandler.
  apply (leaf_refines tenv k params metadata handler handler' body uses
           Hleaf).
  intros index tenv_use args statics hp rd valuation vars conts vs
    handler_vars Hnth Hshows Hshows_use Hargs Hkinds Hbind.
  apply Hhandler.
  pose proof (nth_error_In _ _ Hnth) as Hin.
  apply (closures_union_env_satisfied tenv params metadata uses target
           (tenv_use, args) statics vars vs handler_vars Henv Hin);
    [ | exact (shows_satisfies tenv statics valuation vars Hshows)
      | exact (shows_satisfies tenv_use statics valuation vars Hshows_use)
      | exact Hargs | exact Hkinds | exact Hbind ].
  intros y Hy.
  exact (proj2 (leaf_uses_in_scope tenv k body uses tenv_use args y Hleaf Hin)
           Hy).
Qed.

(* A variable of terms stays one under the environment of the body of
   a [Let]. *)
Lemma let_env_in_scope (tenv tenv' : Typing_env.t) (x : variable)
    (declared : flambda_kind) (d : named) (y : variable) :
  let_env tenv x declared d = Some tenv' ->
  in_scope_in_terms tenv y ->
  in_scope_in_terms tenv' y.
Proof.
  unfold let_env. generalize (resolve_value_slot tenv declared d).
  intros resolved Henv Hy.
  assert (Hdefine :
    ∀ kind : flambda_kind,
      typing_env_well_formed tenv = true ->
      Typing_env.mem None tenv (Name_var x) = false ->
      in_scope_in_terms
        (Typing_env.add_variable_definition tenv x kind Name_mode.normal) y).
  { intros kind Hwf Hfresh.
    apply (add_variable_definition_in_scope tenv x kind y Hwf). right.
    split; [ exact (not_bound_not_in_scope tenv x y Hfresh Hy) | exact Hy ]. }
  destruct (simplify_let tenv x resolved) as [ | d' tenv'' | ] eqn:Hlet;
    [ discriminate | | ].
  - injection Henv as <-. revert Hlet. unfold simplify_let.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Hwf Hfresh].
    apply negb_true_iff in Hfresh.
    destruct (simplify_named tenv resolved) as [ | s | ty | kind | ];
      try discriminate.
    + unfold bind_simple.
      destruct (simplify_simple tenv s) as [[ty canonical] | ];
        [ | discriminate ].
      destruct (simple_in_terms_b tenv canonical
                && simple_of_kind_b tenv canonical
                     (FlambdaTypeGrammar.kind ty));
        [ | discriminate ].
      destruct (Typing_env.add_alias
                  (Typing_env.add_variable_definition tenv x
                     (FlambdaTypeGrammar.kind ty) Name_mode.normal)
                  (Var x) canonical)
        as [[ | result | ] | ] eqn:Hadd; try discriminate.
      intros [= _ <-].
      apply (add_alias_in_scope _ (Var x) canonical result y Hadd).
      exact (Hdefine _ Hwf Hfresh).
    + destruct (type_in_scope_b tenv ty); [ | discriminate ].
      intros [= _ <-]. apply replace_equation_in_scope.
      exact (Hdefine _ Hwf Hfresh).
    + intros [= _ <-]. exact (Hdefine _ Hwf Hfresh).
  - unfold declare_variable in Henv.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Hwf Hfresh].
    apply negb_true_iff in Hfresh. injection Henv as <-.
    exact (Hdefine _ Hwf Hfresh).
Qed.

(* The same under the environment of a handler. *)
Lemma define_variables_checked_keeps_scope
    (definitions : list (variable * flambda_kind)) :
  ∀ (tenv tenv' : Typing_env.t) (y : variable),
    define_variables_checked tenv definitions = Some tenv' ->
    in_scope_in_terms tenv y ->
    in_scope_in_terms tenv' y.
Proof.
  induction definitions as [ | [x kind] definitions IH];
    intros tenv tenv' y Hdefined Hy; cbn [define_variables_checked] in Hdefined.
  - injection Hdefined as <-. exact Hy.
  - destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Hwf Hfresh].
    apply negb_true_iff in Hfresh.
    apply (IH _ tenv' y Hdefined).
    apply (add_variable_definition_in_scope tenv x kind y Hwf). right.
    split; [ exact (not_bound_not_in_scope tenv x y Hfresh Hy) | exact Hy ].
Qed.

Lemma handler_env_keeps_scope (tenv tenv_handler : Typing_env.t)
    (params : list variable) (metadata : handler_metadata) (y : variable) :
  handler_env tenv params metadata = Some tenv_handler ->
  in_scope_in_terms tenv y ->
  in_scope_in_terms tenv_handler y.
Proof.
  unfold handler_env. cbv zeta.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params));
    [ | discriminate ].
  apply define_variables_checked_keeps_scope.
Qed.

(* A variable of terms of the environment of the body is one at every
   application gathered. *)
Fixpoint gather_uses_in_scope (body : expr) {struct body} :
  ∀ (tenv : Typing_env.t) (k : continuation) (h : continuation_handler)
    (uses : list (Typing_env.t * list simple)) (xs : list variable)
    (ks : list continuation),
    gather_uses tenv k h body = Some (uses, xs, ks) ->
    ∀ (use : Typing_env.t * list simple) (y : variable),
      use ∈ uses ->
      in_scope_in_terms tenv y ->
      in_scope_in_terms (fst use) y.
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros tenv k h uses xs ks Hgather; cbn [gather_uses] in Hgather;
    try discriminate.
  - destruct (var_free_in_handler_b x h); [ discriminate | ].
    destruct (let_env tenv x declared d) as [tenv' | ] eqn:Henv;
      [ | discriminate ].
    destruct (gather_uses tenv' k h rest) as [[[uses' xs'] ks'] | ]
      eqn:Hrest; [ | discriminate ].
    injection Hgather as <- _ _. intros use y Hin Hy.
    apply (gather_uses_in_scope rest tenv' k h uses' xs' ks' Hrest use y Hin).
    exact (let_env_in_scope tenv tenv' x declared d y Henv Hy).
  - destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k' h));
      [ | discriminate ].
    destruct (gather_uses tenv k h rest) as [[[uses_rest xs_rest] ks_rest] | ]
      eqn:Hrest; [ | discriminate ].
    pose proof (gather_uses_in_scope rest tenv k h uses_rest xs_rest ks_rest
                  Hrest)
      as IHrest.
    destruct (cont_free_in_handler_b k h').
    + destruct h' as [params' metadata' body'].
      destruct (negb (cont_free_in_handler_b k h)
                && forallb (fun x => negb (var_free_in_handler_b x h))
                     params');
        [ | discriminate ].
      destruct (handler_env tenv params' metadata') as [tenv_handler | ]
        eqn:Henv; [ | discriminate ].
      destruct (gather_uses tenv_handler k h body')
        as [[[uses_h xs_h] ks_h] | ] eqn:Hinner; [ | discriminate ].
      injection Hgather as <- _ _. intros use y Hin Hy.
      apply in_app_or in Hin. destruct Hin as [Hin | Hin].
      * apply (gather_uses_in_scope body' tenv_handler k h uses_h xs_h ks_h
                 Hinner use y Hin).
        exact (handler_env_keeps_scope tenv tenv_handler params' metadata' y
                 Henv Hy).
      * exact (IHrest use y Hin Hy).
    + injection Hgather as <- _ _. exact IHrest.
  - destruct (leaf_uses tenv k (Apply_cont ac)) as [uses' | ] eqn:Hleaf;
      [ | discriminate ].
    injection Hgather as <- _ _. intros [tenv_use args] y Hin Hy.
    exact (proj2
             (leaf_uses_in_scope tenv k _ uses' tenv_use args y Hleaf Hin) Hy).
  - destruct (leaf_uses tenv k (Switch scrutinee arms)) as [uses' | ]
      eqn:Hleaf; [ | discriminate ].
    injection Hgather as <- _ _. intros [tenv_use args] y Hin Hy.
    exact (proj2
             (leaf_uses_in_scope tenv k _ uses' tenv_use args y Hleaf Hin) Hy).
Qed.

Theorem gathered_closures_union_admitted
  : gathered_closures_union_admitted_statement.
Proof.
  intros tenv k params metadata handler body uses xs ks target use Hknown
    Hgather Hkeeps Henv Hin statics vars vs handler_vars Hsat_use Hargs Hkinds
    Hbind.
  pose proof (fun y : variable =>
                gather_uses_in_scope body tenv k _ uses xs ks Hgather use y
                  Hin)
    as Hscope.
  apply (closures_union_env_satisfied tenv params metadata uses target use
           statics vars vs handler_vars Henv Hin Hscope);
    [ | exact Hsat_use | exact Hargs | exact Hkinds | exact Hbind ].
  destruct Hsat_use as [Hbound [valuation [Hsat Hagrees]]]. split.
  - intros y Hy. exact (Hbound y (Hscope y Hy)).
  - exists valuation. split.
    + exact (gather_uses_weaken tenv k _ body uses xs ks use statics
               valuation Hknown Hgather Hkeeps Hin Hsat).
    + intros y Hy. exact (Hagrees y (Hscope y Hy)).
Qed.

Theorem gathered_closures_union_refines
  : gathered_closures_union_refines_statement.
Proof.
  intros tenv k params metadata handler handler' body uses xs ks target
    Hknown Hgather Hkeeps Hfree Henv Hhandler.
  apply (gather_uses_refines tenv k params metadata handler handler' body
           target uses xs ks Hgather Hfree);
    [ | exact Hhandler ].
  intros use Hin.
  exact (gathered_closures_union_admitted tenv k params metadata handler body
           uses xs ks target use Hknown Hgather Hkeeps Henv Hin).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* A case of closures, renamed. *)
Local Abbreviation rename_case r :=
  (rename_closures_case (rename_closures_entry r) (rename_env_extension r)).

Lemma closures_of_head_renamed (r : renaming) (head : head_of_kind_value) :
  closures_of_head (rename_head_of_kind_value r head)
  = option_map (rename_row_like_for_closures r) (closures_of_head head).
Proof.
  destruct head
    as [[ | [is_int immediates get_tag blocks extensions is_unique | alloc
            | k contents alloc | closures alloc | strs]
         | ]
        [ | is_null]];
    reflexivity.
Qed.

Lemma closures_known_renamed (r : renaming) (tenv : Typing_env.t)
    (ty : flambda_type) :
  closures_known (rename_typing_env r tenv) (rename_type r ty)
  = option_map (rename_row_like_for_closures r) (closures_known tenv ty).
Proof.
  unfold closures_known. rewrite expand_head_rename.
  destruct (expand_head tenv ty) as [expanded | ]; [ | reflexivity ].
  cbn [option_map]. unfold rename_expanded. cbn [Expanded_type.descr].
  destruct (Expanded_type.descr expanded)
    as [ | [head | head | k head | head | head] | ];
    try reflexivity.
  exact (closures_of_head_renamed r head).
Qed.

Lemma cases_of_renamed (r : renaming) (closures : row_like_for_closures) :
  cases_of (rename_row_like_for_closures r closures)
  = map_values (rename_case r) (cases_of closures).
Proof. destruct closures as [known_closures]. reflexivity. Qed.

Lemma forallb_same {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, f a = g a) -> forallb f l = forallb g l.
Proof.
  intros Hfg. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hfg, IH. reflexivity.
Qed.

Lemma closures_agree_renamed (r : renaming)
    (closures closures' : row_like_for_closures) :
  closures_agree (rename_row_like_for_closures r closures)
    (rename_row_like_for_closures r closures')
  = closures_agree closures closures'.
Proof.
  unfold closures_agree. rewrite !cases_of_renamed, domain_map_values.
  apply forallb_same. intros slot. rewrite !lookup_map_values.
  destruct (lookup slot (cases_of closures)) as [case | ]; [ | reflexivity ].
  destruct (lookup slot (cases_of closures')) as [case' | ];
    [ | reflexivity ].
  exact (proj1 type_eqb_rename r (one_case_type slot case)
           (one_case_type slot case')).
Qed.

Lemma closures_union_renamed (r : renaming)
    (closures closures' : row_like_for_closures) :
  closures_union (rename_row_like_for_closures r closures)
    (rename_row_like_for_closures r closures')
  = option_map (rename_row_like_for_closures r)
      (closures_union closures closures').
Proof.
  unfold closures_union. rewrite closures_agree_renamed.
  destruct (closures_agree closures closures'); [ | reflexivity ].
  cbn [option_map rename_row_like_for_closures]. rewrite !cases_of_renamed.
  do 2 f_equal. unfold union_with. apply merge_values.
  intros [case | ] [case' | ]; reflexivity.
Qed.

Lemma closures_union_all_renamed (r : renaming)
    (rest : list row_like_for_closures) :
  ∀ first : row_like_for_closures,
    closures_union_all (rename_row_like_for_closures r first)
      (map (rename_row_like_for_closures r) rest)
    = option_map (rename_row_like_for_closures r)
        (closures_union_all first rest).
Proof.
  induction rest as [ | closures rest IH]; intros first;
    cbn [closures_union_all map]; [ reflexivity | ].
  rewrite closures_union_renamed.
  destruct (closures_union first closures) as [union | ]; [ | reflexivity ].
  apply IH.
Qed.

Lemma map_opt_mapped {A A' B B' : Type} (f : A -> option B)
    (f' : A' -> option B') (g : A -> A') (h : B -> B') (l : list A) :
  (∀ a : A, f' (g a) = option_map h (f a)) ->
  map_opt f' (map g l) = option_map (map h) (map_opt f l).
Proof.
  intros Hf. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hf, IH. destruct (f a) as [b | ]; [ | reflexivity ].
  destruct (map_opt f l); reflexivity.
Qed.

Lemma argument_closures_renamed (r : renaming) (index : nat)
    (use : Typing_env.t * list simple) :
  argument_closures index (rename_application r use)
  = option_map (rename_row_like_for_closures r) (argument_closures index use).
Proof.
  unfold argument_closures, rename_application. cbn [fst snd].
  rewrite nth_error_map.
  destruct (nth_error (snd use) index) as [arg | ]; [ | reflexivity ].
  cbn [option_map]. rewrite simplify_simple_equivariant.
  destruct (simplify_simple (fst use) arg) as [[ty canonical] | ];
    [ | reflexivity ].
  cbn [option_map rename_answer fst snd]. apply closures_known_renamed.
Qed.

Lemma union_of_arguments_renamed (r : renaming)
    (uses : list (Typing_env.t * list simple)) (index : nat) :
  union_of_arguments (map (rename_application r) uses) index
  = option_map (rename_type r) (union_of_arguments uses index).
Proof.
  unfold union_of_arguments.
  rewrite (map_opt_mapped (argument_closures index) (argument_closures index)
             (rename_application r) (rename_row_like_for_closures r) uses
             (argument_closures_renamed r index)).
  destruct (map_opt (argument_closures index) uses) as [[ | first rest] | ];
    try reflexivity.
  cbn [option_map map]. rewrite closures_union_all_renamed.
  destruct (closures_union_all first rest); reflexivity.
Qed.

Lemma parameter_union_renamed (r : renaming) (scope : Typing_env.t)
    (uses : list (Typing_env.t * list simple)) (index : nat)
    (kind : flambda_kind) :
  parameter_union (rename_typing_env r scope)
    (map (rename_application r) uses) index kind
  = option_map (rename_type r) (parameter_union scope uses index kind).
Proof.
  unfold parameter_union. destruct kind; try reflexivity.
  rewrite union_of_arguments_renamed.
  destruct (union_of_arguments uses index) as [ty | ]; [ | reflexivity ].
  cbn [option_map]. rewrite type_in_scope_b_renamed.
  destruct (type_in_scope_b scope ty); reflexivity.
Qed.

Lemma parameter_unions_renamed (r : renaming) (scope : Typing_env.t)
    (uses : list (Typing_env.t * list simple)) (kinds : list flambda_kind) :
  ∀ index : nat,
    parameter_unions (rename_typing_env r scope)
      (map (rename_application r) uses) index kinds
    = map (option_map (rename_type r))
        (parameter_unions scope uses index kinds).
Proof.
  induction kinds as [ | kind kinds IH]; intros index;
    cbn [parameter_unions map]; [ reflexivity | ].
  rewrite parameter_union_renamed, IH. reflexivity.
Qed.

(* A definition, renamed. *)
Definition rename_typed_definition (r : renaming)
    (definition : variable * flambda_kind * option flambda_type)
    : variable * flambda_kind * option flambda_type :=
  (rename_variable r (fst (fst definition)), snd (fst definition),
   option_map (rename_type r) (snd definition)).

Lemma define_typed_checked_renamed (r : renaming)
    (definitions : list (variable * flambda_kind * option flambda_type)) :
  ∀ tenv : Typing_env.t,
    define_typed_checked (rename_typing_env r tenv)
      (map (rename_typed_definition r) definitions)
    = option_map (rename_typing_env r)
        (define_typed_checked tenv definitions).
Proof.
  induction definitions as [ | [[x kind] ty] definitions IH]; intros tenv;
    cbn [map define_typed_checked rename_typed_definition fst snd];
    [ reflexivity | ].
  rewrite (typing_env_well_formed_rename r tenv).
  change (Name_var (rename_variable r x)) with (rename_name r (Name_var x)).
  rewrite (mem_equivariant r None tenv (Name_var x)).
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x)));
    [ | reflexivity ].
  cbv zeta.
  rewrite (add_variable_definition_equivariant r tenv x kind
             Name_mode.normal).
  destruct ty as [ty | ]; cbn [option_map]; [ | apply IH ].
  rewrite (replace_equation_equivariant r _ (Name_var x) ty). apply IH.
Qed.

(* The definitions made of renamed parameters and renamed types. *)
Lemma typed_definitions_renamed (r : renaming) (params : list variable) :
  ∀ (kinds : list flambda_kind) (types : list (option flambda_type)),
    combine (combine (map (rename_variable r) params) kinds)
      (map (option_map (rename_type r)) types)
    = map (rename_typed_definition r) (combine (combine params kinds) types).
Proof.
  induction params as [ | param params IH]; intros [ | kind kinds];
    try reflexivity.
  intros [ | ty types]; [ reflexivity | ].
  cbn [map combine]. rewrite IH. reflexivity.
Qed.

Theorem closures_union_env_rename : closures_union_env_rename_statement.
Proof.
  intros r tenv params metadata uses. unfold closures_union_env. cbv zeta.
  rewrite length_map.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params));
    [ | reflexivity ].
  rewrite parameter_unions_renamed, typed_definitions_renamed.
  apply define_typed_checked_renamed.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Theorem example_union_two_slots : example_union_two_slots_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_union_same_case : example_union_same_case_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_union_one_slot_two_codes
  : example_union_one_slot_two_codes_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_union_not_a_closure : example_union_not_a_closure_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_union_slot_in_scope : example_union_slot_in_scope_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_union_slot_out_of_scope
  : example_union_slot_out_of_scope_statement.
Proof. vm_compute. reflexivity. Qed.
