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

(* Proofs of the statements of FlambdaSimplifyStaticsSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaClosuresProofs FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaTypingEnvRefinementProofs
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaProvers FlambdaProversSpec
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPrimitivesProofs
  FlambdaSimplifyPrimitivesRenamingSpec FlambdaSimplifyPrimitivesRenamingProofs
  FlambdaSimplifyProjectionsSpec FlambdaSimplifyProjectionsProofs
  FlambdaSimplifyBoxingProofs FlambdaSimplifyApplySpec
  FlambdaSimplifyApplyProofs FlambdaSimplifyStaticsSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* A symbol has the type of its constant                                    *)
(* ======================================================================== *)

(* The values of arguments have the types of the arguments. *)
Lemma arguments_typed (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (args : list simple) (tys : list flambda_type) :
  env_satisfies tenv statics vars ->
  Forall2 (argument_type tenv) args tys ->
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    Forall2 (type_satisfied statics vars) vs tys.
Proof.
  intros Hsat Htys vs Hvs.
  pose proof (eval_simples_values vars args vs Hvs) as Hvalues.
  clear Hvs. revert vs Hvalues.
  induction Htys as [ | arg ty args tys Hty Htys IH]; intros vs Hvalues;
    inversion Hvalues as [ | arg' v args' vs' Hv Hrest]; subst;
    [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact (IH vs' Hrest) ].
  exact (argument_type_in_machine tenv statics vars arg ty v Hsat Hty Hv).
Qed.

(* Values that have types of the kind value are of the kind value. *)
Lemma kinds_of_values_typed (statics : static_env) (vars : env)
    (vs : list value) (tys : list flambda_type) :
  Forall2 (type_satisfied statics vars) vs tys ->
  forallb (fun ty => kind_eqb (FlambdaTypeGrammar.kind ty) Value) tys = true ->
  map kind_of_value vs = repeat Value (length vs).
Proof.
  intros Htyped. induction Htyped as [ | v ty vs tys Hv Hrest IH]; simpl;
    [ reflexivity | ].
  intros Hkinds. apply andb_true_iff in Hkinds.
  destruct Hkinds as [Hkind Hkinds]. f_equal; [ | exact (IH Hkinds) ].
  rewrite (type_satisfied_kind statics vars v ty Hv).
  destruct (FlambdaTypeGrammar.kind ty); try discriminate. reflexivity.
Qed.

Lemma boxable_number_of_kind_unboxed (kind : flambda_kind)
    (k : boxable_number) :
  boxable_number_of_kind kind = Some k -> kind = unboxed_kind k.
Proof.
  destruct kind as [ | [ | | | | | | | | | | | ] | | ]; simpl;
    try discriminate; intros [= <-]; reflexivity.
Qed.

Theorem static_const_typed : static_const_typed_statement.
Proof.
  intros tenv statics vars s const ty allocated Hsat Htype Heval Hbound.
  unfold static_const_type_in in Htype.
  destruct (simplify_simples tenv (static_const_simples const))
    as [answers | ] eqn:Hanswers; [ | discriminate Htype ].
  simpl in Htype.
  destruct (simplify_simples_answers tenv _ answers Hanswers) as [Htys _].
  clear Hanswers. remember (map fst answers) as tys eqn:Heqtys in *.
  clear Heqtys answers.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    [ | | simpl in Htype, Htys, Heval | simpl in Htype, Htys, Heval
    | simpl in Htype, Htys, Heval | discriminate Htype ].
  - (* A closure that is the only one of its set. *)
    cbn [static_const_type] in Htype. injection Htype as <-.
    cbn [static_const_simples] in Htys. cbn [eval_static_const] in Heval.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs;
      [ | discriminate Heval ].
    simpl in Heval. injection Heval as <-.
    apply (closure_of_set_type_satisfied tenv statics vars
             unnamed_function_slot [(unnamed_function_slot, code)] slots tys
             vs (Value_symbol s) Hsat Htys Hvs).
    cbn [closure_in_set]. rewrite Hbound. reflexivity.
  - (* A closure of a set. *)
    cbn [static_const_type] in Htype. injection Htype as <-.
    cbn [static_const_simples] in Htys. cbn [eval_static_const] in Heval.
    destruct (lookup fslot decls) as [code | ] eqn:Hslot;
      [ | discriminate Heval ].
    simpl in Heval.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs;
      [ | discriminate Heval ].
    simpl in Heval. injection Heval as <-.
    apply (closure_of_set_type_satisfied tenv statics vars fslot decls slots
             tys vs (Value_symbol s) Hsat Htys Hvs).
    cbn [closure_in_set]. rewrite Hbound. simpl. rewrite Hslot. reflexivity.
  - (* A block. *)
    destruct ((tag <? no_scan_tag)
              && forallb
                   (fun ty => kind_eqb (FlambdaTypeGrammar.kind ty) Value)
                   tys) eqn:Hconditions; [ | discriminate Htype ].
    injection Htype as <-. apply andb_true_iff in Hconditions.
    destruct Hconditions as [Htag Hkinds]. apply Z.ltb_lt in Htag.
    destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
      [ | discriminate Heval ].
    simpl in Heval. injection Heval as <-.
    pose proof (arguments_typed tenv statics vars fields tys Hsat Htys vs Hvs)
      as Htyped.
    apply immutable_block_satisfied. exists vs.
    split; [ unfold block_of; rewrite Hbound; reflexivity | ].
    split; [ | exact Htyped ].
    split; [ exact Htag | ].
    exact (kinds_of_values_typed statics vars vs tys Htyped Hkinds).
  - (* A boxed number. *)
    destruct tys as [ | cty [ | cty' tys]]; try discriminate Htype.
    destruct (boxable_number_of_kind (FlambdaTypeGrammar.kind cty))
      as [k | ] eqn:Hk; [ | discriminate Htype ].
    simpl in Htype. injection Htype as <-.
    inversion Htys as [ | arg ty0 args tys0 Hty _]; subst.
    destruct (eval_simple vars contents) as [v | ] eqn:Hv;
      [ | discriminate Heval ].
    simpl in Heval. injection Heval as <-.
    pose proof (argument_type_in_machine tenv statics vars contents cty v
                  Hsat Hty Hv) as Htyped.
    pose proof (type_satisfied_kind statics vars v cty Htyped) as Hkind.
    rewrite (boxable_number_of_kind_unboxed _ k Hk) in Hkind.
    destruct v as [c | | | | | | | | ];
      try (destruct k; discriminate Hkind).
    apply box_number_satisfied. exists c.
    split; [ | split; [ exact Hkind | exact Htyped ] ].
    unfold boxed_number_of. rewrite Hbound.
    unfold boxable_contents, boxable_number_of_const.
    simpl in Hkind. rewrite Hkind.
    destruct k; reflexivity.
  - (* A string. *)
    injection Htype as <-. injection Heval as <-.
    apply this_immutable_string_satisfied. unfold string_of.
    rewrite Hbound. reflexivity.
Qed.

(* A variable of one of the types of arguments is in scope. *)
Lemma argument_types_in_scope (tenv : Typing_env.t) (args : list simple)
    (tys : list flambda_type) (y : variable) :
  Forall2 (argument_type tenv) args tys ->
  ∀ ty : flambda_type,
    ty ∈ tys -> y ∈ free_variables ty -> in_scope_in_terms tenv y.
Proof.
  intros Htys ty Hin Hfree.
  induction Htys as [ | arg ty' args tys Hty Htys IH]; [ contradiction | ].
  destruct Hin as [<- | Hin];
    [ exact (argument_type_in_scope tenv arg ty' y Hty Hfree)
    | exact (IH Hin) ].
Qed.

Theorem static_const_type_in_scope : static_const_type_in_scope_statement.
Proof.
  intros tenv const ty y Htype Hy.
  unfold static_const_type_in in Htype.
  destruct (simplify_simples tenv (static_const_simples const))
    as [answers | ] eqn:Hanswers; [ | discriminate Htype ].
  simpl in Htype.
  destruct (simplify_simples_answers tenv _ answers Hanswers) as [Htys _].
  clear Hanswers. remember (map fst answers) as tys eqn:Heqtys in *.
  clear Heqtys answers.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    [ | | simpl in Htype, Htys | simpl in Htype, Htys
    | simpl in Htype, Htys | discriminate Htype ].
  - cbn [static_const_type] in Htype. injection Htype as <-.
    cbn [static_const_simples] in Htys.
    destruct
      (closure_of_set_type_free_variables unnamed_function_slot
         [(unnamed_function_slot, code)] (map fst slots) tys y Hy)
      as [ty [Hin Hfree]].
    exact (argument_types_in_scope tenv _ tys y Htys ty Hin Hfree).
  - cbn [static_const_type] in Htype. injection Htype as <-.
    cbn [static_const_simples] in Htys.
    destruct
      (closure_of_set_type_free_variables fslot decls (map fst slots) tys y
         Hy)
      as [ty [Hin Hfree]].
    exact (argument_types_in_scope tenv _ tys y Htys ty Hin Hfree).
  - destruct ((tag <? no_scan_tag) && _); [ | discriminate Htype ].
    injection Htype as <-.
    destruct (free_variables_make_block_type tag tys y Hy)
      as [ty [Hin Hfree]].
    clear Hy. induction Htys as [ | arg ty' args tys Hty Htys IH];
      [ contradiction | ].
    destruct Hin as [<- | Hin];
      [ exact (argument_type_in_scope tenv arg ty' y Hty Hfree)
      | exact (IH Hin) ].
  - destruct tys as [ | cty [ | cty' tys]]; try discriminate Htype.
    destruct (boxable_number_of_kind (FlambdaTypeGrammar.kind cty))
      as [k | ]; [ | discriminate Htype ].
    simpl in Htype. injection Htype as <-.
    inversion Htys as [ | arg ty0 args tys0 Hty _]; subst.
    exact (argument_type_in_scope tenv contents cty y Hty
      (box_number_free_variables k cty Alloc_mode.Heap y Hy)).
  - injection Htype as <-. contradiction.
Qed.

(* ======================================================================== *)
(* The environment follows the term                                         *)
(* ======================================================================== *)

Lemma add_symbol_definition_env_satisfies_strictly (tenv : Typing_env.t)
    (s : symbol) (statics : static_env) (vars : env) :
  env_satisfies_strictly tenv statics vars ->
  env_satisfies_strictly (Typing_env.add_symbol_definition tenv s) statics
    vars.
Proof.
  intros [Hbound [valuation [Hsatisfied Hagrees]]].
  split; [ exact Hbound | ]. exists valuation.
  split; [ | exact Hagrees ].
  exact (proj2 (add_symbol_definition_satisfied _ statics valuation tenv s)
    Hsatisfied).
Qed.

Lemma add_symbol_definitions_env_satisfies_strictly (tenv : Typing_env.t)
    (syms : list symbol) (statics : static_env) (vars : env) :
  env_satisfies_strictly tenv statics vars ->
  env_satisfies_strictly (Typing_env.add_symbol_definitions tenv syms)
    statics vars.
Proof.
  intros [Hbound [valuation [Hsatisfied Hagrees]]].
  split; [ exact Hbound | ]. exists valuation.
  split; [ | exact Hagrees ].
  clear Hbound Hagrees. induction syms as [ | sym syms IH].
  - destruct tenv. exact Hsatisfied.
  - exact (proj2 (add_symbol_definition_satisfied _ statics valuation
                    (Typing_env.add_symbol_definitions tenv syms) sym) IH).
Qed.

(* The variables in scope once a symbol is given a type. *)
Lemma replace_symbol_equation_in_scope (tenv : Typing_env.t) (s : symbol)
    (ty : flambda_type) (y : variable) :
  in_scope_in_terms (Typing_env.replace_equation tenv (Name_symbol s) ty) y
  <-> in_scope_in_terms tenv y.
Proof.
  rewrite !in_scope_in_terms_lookup.
  change (Typing_env.min_binding_time
            (Typing_env.replace_equation tenv (Name_symbol s) ty))
    with (Typing_env.min_binding_time tenv).
  rewrite (replace_equation_lookup_other tenv (Name_symbol s) (Name_var y) ty);
    [ reflexivity | discriminate ].
Qed.

(* A symbol is given a type that the pointer which it is has. *)
Lemma replace_symbol_equation_env_satisfies_strictly (tenv : Typing_env.t)
    (statics : static_env) (vars : env) (s : symbol) (ty : flambda_type) :
  (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
  type_satisfied statics vars (Value_symbol s) ty ->
  env_satisfies_strictly tenv statics vars ->
  env_satisfies_strictly (Typing_env.replace_equation tenv (Name_symbol s) ty)
    statics vars.
Proof.
  intros Hfree Htype [Hbound [valuation [Hsatisfied Hagrees]]].
  split.
  - intros y Hy. apply Hbound.
    apply (replace_symbol_equation_in_scope tenv s ty y). exact Hy.
  - exists valuation. split.
    + apply replace_equation_satisfied; [ exact Hsatisfied | ].
      intros v Hv. right. injection Hv as <-.
      apply (type_satisfied_coincide statics vars valuation (Value_symbol s)
               ty);
        [ | exact Htype ].
      intros y Hin. symmetry. exact (Hagrees y (Hfree y Hin)).
    + intros y Hy. apply Hagrees.
      apply (replace_symbol_equation_in_scope tenv s ty y). exact Hy.
Qed.

Theorem symbol_typed_env_satisfies : symbol_typed_env_satisfies_statement.
Proof.
  intros tenv statics vars s const ty allocated Htype Heval Hbound Hsat.
  unfold define_symbol_with_type.
  apply replace_symbol_equation_env_satisfies_strictly.
  - intros y Hy.
    exact (static_const_type_in_scope tenv const ty y Htype Hy).
  - exact (static_const_typed tenv statics vars s const ty allocated
      (env_satisfies_strictly_env_satisfies tenv statics vars Hsat) Htype
      Heval Hbound).
  - apply add_symbol_definition_env_satisfies_strictly. exact Hsat.
Qed.

(* The symbols of a group that are not bound to closures, in a static
   environment that has the whole group bound. *)
Lemma add_block_like_bindings_env_satisfies (statics extended : static_env)
    (vars : env) (all : list static_binding) :
  eval_static_bindings vars all statics = Some extended ->
  ∀ (bindings : list static_binding) (tenv : Typing_env.t),
    (∀ binding : static_binding, binding ∈ bindings -> binding ∈ all) ->
    env_satisfies_strictly tenv extended vars ->
    env_satisfies_strictly (add_block_like_bindings tenv bindings) extended
      vars.
Proof.
  intros Heval.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros tenv Hall Hsat; simpl.
  - exact Hsat.
  - apply IH; [ intros b Hb; apply Hall; right; exact Hb | exact Hsat ].
  - apply IH; [ intros b Hb; apply Hall; right; exact Hb | exact Hsat ].
  - apply IH; [ intros b Hb; apply Hall; right; exact Hb | ].
    destruct (is_closure_const const); [ exact Hsat | ].
    destruct (static_const_type_in tenv const) as [ty | ] eqn:Htype;
      [ | apply add_symbol_definition_env_satisfies_strictly; exact Hsat ].
    destruct
      (eval_static_bindings_binds vars all statics extended s const Heval
         (Hall _ (or_introl eq_refl)))
      as (allocated & Hconst & Hbound).
    exact (symbol_typed_env_satisfies tenv extended vars s const ty allocated
      Htype Hconst Hbound Hsat).
Qed.

(* The symbols of a group that are bound to closures, in a static
   environment that has the whole group bound. The types are taken in
   an environment that the state satisfies and that has the same
   variables in scope. *)
Lemma add_closure_bindings_env_satisfies (statics extended : static_env)
    (vars : env) (all : list static_binding) (prior : Typing_env.t) :
  eval_static_bindings vars all statics = Some extended ->
  env_satisfies_strictly prior extended vars ->
  ∀ (bindings : list static_binding) (tenv : Typing_env.t),
    (∀ binding : static_binding, binding ∈ bindings -> binding ∈ all) ->
    (∀ y : variable, in_scope_in_terms prior y -> in_scope_in_terms tenv y) ->
    env_satisfies_strictly tenv extended vars ->
    env_satisfies_strictly (add_closure_bindings prior tenv bindings)
      extended vars.
Proof.
  intros Heval Hprior.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros tenv Hall Hscope Hsat; cbn [add_closure_bindings].
  - exact Hsat.
  - apply IH;
      [ intros b Hb; apply Hall; right; exact Hb | exact Hscope | exact Hsat ].
  - apply IH;
      [ intros b Hb; apply Hall; right; exact Hb | exact Hscope | exact Hsat ].
  - destruct (is_closure_const const);
      [ | apply IH;
          [ intros b Hb; apply Hall; right; exact Hb | exact Hscope
          | exact Hsat ] ].
    destruct (static_const_type_in prior const) as [ty | ] eqn:Htype;
      [ | apply IH;
          [ intros b Hb; apply Hall; right; exact Hb | exact Hscope
          | exact Hsat ] ].
    destruct
      (eval_static_bindings_binds vars all statics extended s const Heval
         (Hall _ (or_introl eq_refl)))
      as (allocated & Hconst & Hbound).
    apply IH; [ intros b Hb; apply Hall; right; exact Hb | | ].
    + intros y Hy.
      apply (replace_symbol_equation_in_scope tenv s ty y).
      exact (Hscope y Hy).
    + apply replace_symbol_equation_env_satisfies_strictly;
        [ | | exact Hsat ].
      * intros y Hy. apply Hscope.
        exact (static_const_type_in_scope prior const ty y Htype Hy).
      * exact (static_const_typed prior extended vars s const ty allocated
          (env_satisfies_strictly_env_satisfies prior extended vars Hprior)
          Htype Hconst Hbound).
Qed.

Theorem add_static_bindings_env_satisfies :
  add_static_bindings_env_satisfies_statement.
Proof.
  intros tenv statics extended vars bindings Heval Hsat.
  unfold add_static_bindings. cbv zeta.
  assert (Hprior :
    env_satisfies_strictly
      (add_block_like_bindings
         (Typing_env.add_symbol_definitions tenv
            (closure_symbols_of bindings))
         bindings)
      extended vars);
    [ | exact (add_closure_bindings_env_satisfies statics extended vars
                 bindings _ Heval Hprior bindings _
                 (fun binding Hin => Hin) (fun y Hy => Hy) Hprior) ].
  apply (add_block_like_bindings_env_satisfies statics extended vars bindings
           Heval);
    [ intros binding Hin; exact Hin | ].
  apply add_symbol_definitions_env_satisfies_strictly.
  exact (env_satisfies_strictly_statics_extend tenv statics extended vars
    (eval_static_bindings_extend vars bindings statics extended Heval) Hsat).
Qed.

Theorem let_static_typed : let_static_typed_statement.
Proof.
  intros tenv bindings body body' Hbody vars conts statics hp rd Hsat.
  destruct (eval_static_bindings vars bindings statics) as [extended | ]
    eqn:Hbindings.
  - assert (Hstep :
      ∀ b : expr,
        step (State (Let_static bindings b) vars conts statics hp rd)
        = Step (State b vars conts extended hp rd)).
    { intros b. rewrite step_let_static, Hbindings. reflexivity. }
    eapply refines_step; [ apply Hstep | apply Hstep | ].
    apply Hbody.
    exact (add_static_bindings_env_satisfies tenv statics extended vars
      bindings Hbindings Hsat).
  - apply step_undefined_refines. rewrite step_let_static, Hbindings.
    reflexivity.
Qed.

(* ======================================================================== *)
(* The symbols defined                                                      *)
(* ======================================================================== *)

Lemma in_closure_symbols_of (bindings : list static_binding) (s : symbol) :
  s ∈ closure_symbols_of bindings
  <-> ∃ const : static_const,
        Bind_symbol s const ∈ bindings ∧ is_closure_const const = true.
Proof.
  induction bindings as [ | [id c | id | s' const] bindings IH]; simpl.
  - split; [ intros [] | intros [const [[] _]] ].
  - rewrite IH. split; intros [const [Hin Hc]]; exists const;
      [ auto | destruct Hin as [[=] | Hin]; auto ].
  - rewrite IH. split; intros [const [Hin Hc]]; exists const;
      [ auto | destruct Hin as [[=] | Hin]; auto ].
  - destruct (is_closure_const const) eqn:Hconst; simpl; rewrite IH.
    + split.
      * intros [<- | [const' [Hin Hc]]];
          [ exists const; auto | exists const'; auto ].
      * intros [const' [[[= <- <-] | Hin] Hc]];
          [ left; reflexivity | right; exists const'; auto ].
    + split.
      * intros [const' [Hin Hc]]. exists const'. auto.
      * intros [const' [[[= <- <-] | Hin] Hc]];
          [ congruence | exists const'; auto ].
Qed.

Lemma defined_symbols_block_like (bindings : list static_binding) :
  ∀ (tenv : Typing_env.t) (s : symbol),
    s ∈ Typing_env.defined_symbols (add_block_like_bindings tenv bindings)
    <-> s ∈ Typing_env.defined_symbols tenv
        ∨ ∃ const : static_const,
            Bind_symbol s const ∈ bindings ∧ is_closure_const const = false.
Proof.
  induction bindings as [ | [id c | id | s' const] bindings IH];
    intros tenv s; simpl.
  - split; [ auto | intros [Hin | [const [[] _]]]; exact Hin ].
  - rewrite IH. split; (intros [Hin | [const [Hin Hc]]]; [ auto | ]);
      right; exists const; [ auto | destruct Hin as [[=] | Hin]; auto ].
  - rewrite IH. split; (intros [Hin | [const [Hin Hc]]]; [ auto | ]);
      right; exists const; [ auto | destruct Hin as [[=] | Hin]; auto ].
  - rewrite IH. destruct (is_closure_const const) eqn:Hconst.
    + split; (intros [Hin | [const' [Hin Hc]]]; [ auto | ]);
        right; exists const'; [ auto | ].
      destruct Hin as [[= <- <-] | Hin]; [ congruence | auto ].
    + assert (Hdefined :
        s ∈ Typing_env.defined_symbols
              match static_const_type_in tenv const with
              | Some ty => define_symbol_with_type tenv s' ty
              | None => Typing_env.add_symbol_definition tenv s'
              end
        <-> s' = s ∨ s ∈ Typing_env.defined_symbols tenv)
        by (destruct (static_const_type_in tenv const); reflexivity).
      rewrite Hdefined. split.
      * intros [[<- | Hin] | [const' [Hin Hc]]];
          [ right; exists const; auto | auto | right; exists const'; auto ].
      * intros [Hin | [const' [[[= <- <-] | Hin] Hc]]];
          [ auto | auto | right; exists const'; auto ].
Qed.

(* Giving the closures their types defines no symbol. *)
Lemma defined_symbols_closure (prior : Typing_env.t)
    (bindings : list static_binding) :
  ∀ tenv : Typing_env.t,
    Typing_env.defined_symbols (add_closure_bindings prior tenv bindings)
    = Typing_env.defined_symbols tenv.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; intros tenv;
    cbn [add_closure_bindings]; [ reflexivity | apply IH | apply IH | ].
  rewrite IH.
  destruct (is_closure_const const); [ | reflexivity ].
  destruct (static_const_type_in prior const); reflexivity.
Qed.

Theorem add_static_bindings_defined_symbols :
  add_static_bindings_defined_symbols_statement.
Proof.
  intros tenv bindings s. unfold add_static_bindings. cbv zeta.
  rewrite defined_symbols_closure, defined_symbols_block_like.
  change
    (Typing_env.defined_symbols
       (Typing_env.add_symbol_definitions tenv (closure_symbols_of bindings)))
    with (closure_symbols_of bindings ++ Typing_env.defined_symbols tenv).
  rewrite in_app_iff, in_closure_symbols_of. split.
  - intros [[[const [Hin _]] | Hin] | [const [Hin _]]];
      [ right; exists const; exact Hin | left; exact Hin
      | right; exists const; exact Hin ].
  - intros [Hin | [const Hin]]; [ left; right; exact Hin | ].
    destruct (is_closure_const const) eqn:Hconst;
      [ left; left; exists const; auto | right; exists const; auto ].
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Lemma static_const_simples_renamed (r : renaming) (const : static_const) :
  static_const_simples (rename_static_const r const)
  = map (rename_simple r) (static_const_simples const).
Proof.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl; try reflexivity; rewrite !map_map; reflexivity.
Qed.

Lemma map_fst_renamed_slots (r : renaming)
    (slots : list (value_slot * simple)) :
  map fst (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = map fst slots.
Proof. rewrite map_map. reflexivity. Qed.

Lemma static_const_type_renamed (r : renaming) (const : static_const)
    (tys : list flambda_type) :
  static_const_type (rename_static_const r const) (map (rename_type r) tys)
  = option_map (rename_type r) (static_const_type const tys).
Proof.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    cbn [static_const_type rename_static_const]; try reflexivity.
  - rewrite map_fst_renamed_slots. cbn [option_map]. f_equal. symmetry.
    exact (closure_of_set_type_rename r unnamed_function_slot
             [(unnamed_function_slot, code)] (map fst slots) tys).
  - rewrite map_fst_renamed_slots. cbn [option_map]. f_equal. symmetry.
    exact (closure_of_set_type_rename r fslot decls (map fst slots) tys).
  - assert (Hkinds :
      forallb (fun ty => kind_eqb (FlambdaTypeGrammar.kind ty) Value)
        (map (rename_type r) tys)
      = forallb (fun ty => kind_eqb (FlambdaTypeGrammar.kind ty) Value) tys).
    { induction tys as [ | ty tys IH]; simpl; [ reflexivity | ].
      rewrite kind_rename_type, IH. reflexivity. }
    rewrite Hkinds.
    destruct ((tag <? no_scan_tag) && _); [ | reflexivity ].
    cbn [option_map]. unfold make_block_type.
    rewrite rename_type_immutable_block. reflexivity.
  - destruct tys as [ | ty [ | ty' tys]]; try reflexivity.
    cbn [map]. rewrite kind_rename_type.
    destruct (boxable_number_of_kind (FlambdaTypeGrammar.kind ty));
      [ | reflexivity ].
    cbn beta iota. cbn [option_map]. rewrite rename_type_box_number.
    reflexivity.
Qed.

Theorem static_const_type_in_rename : static_const_type_in_rename_statement.
Proof.
  intros r tenv const. unfold static_const_type_in.
  rewrite static_const_simples_renamed,
    (simplify_simples_equivariant r tenv (static_const_simples const)).
  destruct (simplify_simples tenv (static_const_simples const))
    as [answers | ];
    [ | reflexivity ].
  simpl.
  replace (map fst (map (rename_answer r) answers))
    with (map (rename_type r) (map fst answers))
    by (rewrite !map_map; reflexivity).
  apply static_const_type_renamed.
Qed.

Lemma is_closure_const_renamed (r : renaming) (const : static_const) :
  is_closure_const (rename_static_const r const) = is_closure_const const.
Proof. destruct const; reflexivity. Qed.

Lemma closure_symbols_of_renamed (r : renaming)
    (bindings : list static_binding) :
  closure_symbols_of (map (rename_static_binding r) bindings)
  = closure_symbols_of bindings.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl;
    try exact IH; [ reflexivity | ].
  rewrite is_closure_const_renamed, IH. reflexivity.
Qed.

Lemma add_block_like_bindings_renamed (r : renaming)
    (bindings : list static_binding) :
  ∀ tenv : Typing_env.t,
    add_block_like_bindings (rename_typing_env r tenv)
      (map (rename_static_binding r) bindings)
    = rename_typing_env r (add_block_like_bindings tenv bindings).
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; intros tenv;
    simpl; try apply IH; [ reflexivity | ].
  rewrite is_closure_const_renamed.
  destruct (is_closure_const const); [ apply IH | ].
  rewrite (static_const_type_in_rename r tenv const).
  destruct (static_const_type_in tenv const) as [ty | ]; simpl.
  - unfold define_symbol_with_type.
    rewrite (add_symbol_definition_equivariant r tenv s).
    change (Name_symbol s) with (rename_name r (Name_symbol s)) at 1.
    rewrite (replace_equation_equivariant r _ (Name_symbol s) ty).
    apply IH.
  - rewrite (add_symbol_definition_equivariant r tenv s). apply IH.
Qed.

Lemma add_closure_bindings_renamed (r : renaming) (prior : Typing_env.t)
    (bindings : list static_binding) :
  ∀ tenv : Typing_env.t,
    add_closure_bindings (rename_typing_env r prior)
      (rename_typing_env r tenv) (map (rename_static_binding r) bindings)
    = rename_typing_env r (add_closure_bindings prior tenv bindings).
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; intros tenv;
    cbn [map rename_static_binding add_closure_bindings];
    [ reflexivity | apply IH | apply IH | ].
  rewrite is_closure_const_renamed.
  destruct (is_closure_const const); [ | apply IH ].
  rewrite (static_const_type_in_rename r prior const).
  destruct (static_const_type_in prior const) as [ty | ]; cbn [option_map];
    [ | apply IH ].
  change (Name_symbol s) with (rename_name r (Name_symbol s)) at 1.
  rewrite (replace_equation_equivariant r tenv (Name_symbol s) ty).
  apply IH.
Qed.

Theorem add_static_bindings_rename : add_static_bindings_rename_statement.
Proof.
  intros r tenv bindings. unfold add_static_bindings. cbv zeta.
  rewrite closure_symbols_of_renamed,
    (add_symbol_definitions_equivariant r tenv (closure_symbols_of bindings)),
    add_block_like_bindings_renamed.
  apply add_closure_bindings_renamed.
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem example_load_of_static_block : example_load_of_static_block_statement.
Proof.
  intros k bindings. apply let_static_typed.
  eapply expr_refines_under_trans.
  - (* The load is the [Simple] that the prover answers. *)
    match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_block_load tenv example_q Value
                    (Block_access_values (Some 0) (Some 2) Any_value)
                    Immutable 0 (Symbol example_s)
                    (alias_type_of Value (Symbol example_s)) body) as Hload
    end.
    cbv zeta in Hload.
    match type of Hload with
    | context [Provers.meet_block_field_simple ?tenv ?mode ?kind ?ty ?i] =>
      let answer :=
        eval vm_compute in
          (Provers.meet_block_field_simple tenv mode kind ty i)
      in
      replace (Provers.meet_block_field_simple tenv mode kind ty i)
        with answer in Hload by (vm_compute; reflexivity)
    end.
    cbv beta iota in Hload. apply Hload; [ discriminate | ].
    argument_computed.
  - (* The variable of the load is made equal to it. *)
    eapply (let_simple_refines_under _ example_q Value (Var example_x) _ _
              Value);
      [ vm_compute; reflexivity | vm_compute; reflexivity
      | intros ? [= <-]; vm_compute; reflexivity
      | eexists; vm_compute; reflexivity | vm_compute; reflexivity
      | apply expr_equivalent_under_refines; apply apply_cont_canonical;
        apply Forall2_cons; [ canonical_computed | apply Forall2_nil ] ].
Qed.

(* ======================================================================== *)
(* The environment made is well formed                                      *)
(* ======================================================================== *)

(* The type of a constant is of the kind value, is not an alias, and is
   not the unknown type. *)
Lemma static_const_type_shape (const : static_const)
    (tys : list flambda_type) (ty : flambda_type) :
  static_const_type const tys = Some ty ->
  FlambdaTypeGrammar.kind ty = Value
  ∧ get_alias_opt ty = None
  ∧ is_obviously_unknown ty = false.
Proof.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    cbn [static_const_type].
  - intros [= <-]. repeat split; reflexivity.
  - intros [= <-]. repeat split; reflexivity.
  - destruct ((tag <? no_scan_tag) && _); [ | discriminate ].
    intros [= <-]. repeat split; reflexivity.
  - destruct tys as [ | cty [ | cty' tys]]; try discriminate.
    destruct (boxable_number_of_kind (FlambdaTypeGrammar.kind cty))
      as [k | ]; [ | discriminate ].
    simpl. intros [= <-]. repeat split; reflexivity.
  - intros [= <-]. repeat split; reflexivity.
  - discriminate.
Qed.

Lemma static_const_type_in_shape (tenv : Typing_env.t) (const : static_const)
    (ty : flambda_type) :
  static_const_type_in tenv const = Some ty ->
  FlambdaTypeGrammar.kind ty = Value
  ∧ get_alias_opt ty = None
  ∧ is_obviously_unknown ty = false.
Proof.
  unfold static_const_type_in.
  destruct (simplify_simples tenv (static_const_simples const))
    as [answers | ]; [ | discriminate ].
  simpl. apply static_const_type_shape.
Qed.

(* A variable that may occur in terms is bound. *)
Lemma in_scope_mem (tenv : Typing_env.t) (x : variable) :
  in_scope_in_terms tenv x -> Typing_env.mem None tenv (Name_var x) = true.
Proof.
  unfold in_scope_in_terms, Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv)) as [[ty btm] | ];
    [ intros _; reflexivity | discriminate ].
Qed.

(* Every symbol that has a type either was defined in [tenv0] or has a
   type of the kind value. *)
Definition symbols_of_kind_value (tenv0 tenv : Typing_env.t) : Prop :=
  ∀ (s : symbol) (ty : flambda_type)
    (btm : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_symbol s) (Typing_env.names_to_types tenv)
    = Some (ty, btm) ->
    among s (Typing_env.defined_symbols tenv0) = true
    ∨ FlambdaTypeGrammar.kind ty = Value.

Lemma symbols_of_kind_value_replace (tenv0 tenv : Typing_env.t) (s : symbol)
    (ty : flambda_type) :
  FlambdaTypeGrammar.kind ty = Value ->
  symbols_of_kind_value tenv0 tenv ->
  symbols_of_kind_value tenv0
    (Typing_env.replace_equation tenv (Name_symbol s) ty).
Proof.
  intros Hvalue Hkinds s' ty' btm Hlookup.
  destruct (Pos.eq_dec s' s) as [-> | Hne].
  - rewrite (replace_equation_lookup tenv (Name_symbol s) ty) in Hlookup.
    injection Hlookup as <- _. right. exact Hvalue.
  - rewrite (replace_equation_lookup_other tenv (Name_symbol s)
               (Name_symbol s') ty) in Hlookup;
      [ exact (Hkinds s' ty' btm Hlookup) | congruence ].
Qed.

(* One symbol that is defined, and that the group binds, is given the
   type of its constant, taken in [prior]. *)
Lemma replace_symbol_equation_invariant (tenv0 prior tenv : Typing_env.t)
    (s : symbol) (const : static_const) (ty : flambda_type) :
  among s (Typing_env.defined_symbols tenv0) = false ->
  typing_env_invariant tenv ->
  symbols_of_kind_value tenv0 tenv ->
  among s (Typing_env.defined_symbols tenv) = true ->
  static_const_type_in prior const = Some ty ->
  (∀ y : variable,
    in_scope_in_terms prior y ->
    Typing_env.mem None tenv (Name_var y) = true) ->
  typing_env_invariant (Typing_env.replace_equation tenv (Name_symbol s) ty)
  ∧ symbols_of_kind_value tenv0
      (Typing_env.replace_equation tenv (Name_symbol s) ty).
Proof.
  intros Hfresh Hinvariant Hkinds Hdefined Htype Hscope.
  destruct (static_const_type_in_shape prior const ty Htype)
    as (Hvalue & Halias & Hunknown).
  split; [ | exact (symbols_of_kind_value_replace tenv0 tenv s ty Hvalue
                      Hkinds) ].
  apply replace_equation_invariant.
  - exact Hinvariant.
  - exact Hdefined.
  - intros x Hx. apply Hscope.
    exact (static_const_type_in_scope prior const ty x Htype Hx).
  - intros ty' btm Hlookup. rewrite Hvalue.
    destruct (Hkinds s ty' btm Hlookup) as [Hold | Hkind];
      [ congruence | symmetry; exact Hkind ].
  - intros alias Halias'. rewrite Halias in Halias'. discriminate Halias'.
  - left. exact Hunknown.
Qed.

Lemma add_block_like_bindings_invariant (tenv0 : Typing_env.t)
    (all : list static_binding) :
  (∀ (s : symbol) (const : static_const),
    Bind_symbol s const ∈ all ->
    among s (Typing_env.defined_symbols tenv0) = false) ->
  ∀ (bindings : list static_binding) (tenv : Typing_env.t),
    (∀ binding : static_binding, binding ∈ bindings -> binding ∈ all) ->
    typing_env_invariant tenv ->
    symbols_of_kind_value tenv0 tenv ->
    typing_env_invariant (add_block_like_bindings tenv bindings)
    ∧ symbols_of_kind_value tenv0 (add_block_like_bindings tenv bindings).
Proof.
  intros Hfresh.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros tenv Hall Hinvariant Hkinds; cbn [add_block_like_bindings].
  - split; assumption.
  - apply IH; [ intros b Hb; apply Hall; right; exact Hb | | ]; assumption.
  - apply IH; [ intros b Hb; apply Hall; right; exact Hb | | ]; assumption.
  - destruct (is_closure_const const);
      [ apply IH; [ intros b Hb; apply Hall; right; exact Hb | | ];
        assumption | ].
    assert (Hdefined :
      typing_env_invariant (Typing_env.add_symbol_definition tenv s))
      by (apply add_symbol_definition_invariant; exact Hinvariant).
    destruct (static_const_type_in tenv const) as [ty | ] eqn:Htype.
    + destruct
        (replace_symbol_equation_invariant tenv0 tenv
           (Typing_env.add_symbol_definition tenv s) s const ty
           (Hfresh s const (Hall _ (or_introl eq_refl))) Hdefined Hkinds)
        as [Hinvariant' Hkinds'].
      * change (among s (s :: Typing_env.defined_symbols tenv) = true).
        rewrite among_cons, Pos.eqb_refl. reflexivity.
      * exact Htype.
      * intros y Hy. exact (in_scope_mem tenv y Hy).
      * apply IH; [ intros b Hb; apply Hall; right; exact Hb | | ];
          assumption.
    + apply IH; [ intros b Hb; apply Hall; right; exact Hb | | ];
        assumption.
Qed.

Lemma add_closure_bindings_invariant (tenv0 prior : Typing_env.t)
    (all : list static_binding) :
  (∀ (s : symbol) (const : static_const),
    Bind_symbol s const ∈ all ->
    among s (Typing_env.defined_symbols tenv0) = false) ->
  ∀ (bindings : list static_binding) (tenv : Typing_env.t),
    (∀ binding : static_binding, binding ∈ bindings -> binding ∈ all) ->
    typing_env_invariant tenv ->
    symbols_of_kind_value tenv0 tenv ->
    (∀ y : variable,
      in_scope_in_terms prior y ->
      Typing_env.mem None tenv (Name_var y) = true) ->
    (∀ (s : symbol) (const : static_const),
      Bind_symbol s const ∈ bindings ->
      is_closure_const const = true ->
      among s (Typing_env.defined_symbols tenv) = true) ->
    typing_env_invariant (add_closure_bindings prior tenv bindings).
Proof.
  intros Hfresh.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros tenv Hall Hinvariant Hkinds Hscope Hdefined;
    cbn [add_closure_bindings].
  - exact Hinvariant.
  - apply IH; try assumption;
      [ intros b Hb; apply Hall; right; exact Hb
      | intros s' const' Hin; apply Hdefined; right; exact Hin ].
  - apply IH; try assumption;
      [ intros b Hb; apply Hall; right; exact Hb
      | intros s' const' Hin; apply Hdefined; right; exact Hin ].
  - destruct (is_closure_const const) eqn:Hclosure;
      [ | apply IH; try assumption;
          [ intros b Hb; apply Hall; right; exact Hb
          | intros s' const' Hin; apply Hdefined; right; exact Hin ] ].
    destruct (static_const_type_in prior const) as [ty | ] eqn:Htype;
      [ | apply IH; try assumption;
          [ intros b Hb; apply Hall; right; exact Hb
          | intros s' const' Hin; apply Hdefined; right; exact Hin ] ].
    destruct
      (replace_symbol_equation_invariant tenv0 prior tenv s const ty
         (Hfresh s const (Hall _ (or_introl eq_refl))) Hinvariant Hkinds
         (Hdefined s const (or_introl eq_refl) Hclosure) Htype Hscope)
      as [Hinvariant' Hkinds'].
    apply IH; try assumption.
    + intros b Hb. apply Hall. right. exact Hb.
    + intros s' const' Hin Hclosure'.
      exact (Hdefined s' const' (or_intror Hin) Hclosure').
Qed.

Lemma add_symbol_definitions_invariant (tenv : Typing_env.t)
    (syms : list symbol) :
  typing_env_invariant tenv ->
  typing_env_invariant (Typing_env.add_symbol_definitions tenv syms).
Proof.
  intros Hinvariant. induction syms as [ | sym syms IH].
  - destruct tenv. exact Hinvariant.
  - exact (add_symbol_definition_invariant
             (Typing_env.add_symbol_definitions tenv syms) sym IH).
Qed.

Theorem add_static_bindings_invariant
    : add_static_bindings_invariant_statement.
Proof.
  intros tenv bindings Hinvariant Hfresh. unfold add_static_bindings.
  cbv zeta.
  assert (Hkinds :
    symbols_of_kind_value tenv
      (Typing_env.add_symbol_definitions tenv (closure_symbols_of bindings))).
  { intros s ty btm Hlookup. left.
    pose proof Hinvariant as Hparts. invariant_parts Hparts.
    exact (Hsymbols s ty btm Hlookup). }
  destruct
    (add_block_like_bindings_invariant tenv bindings Hfresh bindings _
       (fun binding Hin => Hin)
       (add_symbol_definitions_invariant tenv (closure_symbols_of bindings)
          Hinvariant)
       Hkinds)
    as [Hprior Hkinds_prior].
  apply (add_closure_bindings_invariant tenv _ bindings Hfresh bindings _
           (fun binding Hin => Hin) Hprior Hkinds_prior
           (fun y Hy => in_scope_mem _ y Hy)).
  intros s const Hin Hclosure. apply among_In.
  apply defined_symbols_block_like. left.
  change
    (s ∈ closure_symbols_of bindings ++ Typing_env.defined_symbols tenv).
  apply in_or_app. left. apply in_closure_symbols_of. exists const.
  split; assumption.
Qed.

Theorem add_static_bindings_well_formed
    : add_static_bindings_well_formed_statement.
Proof.
  intros tenv bindings Hinvariant Hfresh.
  pose proof (add_static_bindings_invariant tenv bindings Hinvariant Hfresh)
    as Hafter.
  split; [ exact (proj1 Hafter) | ].
  exact (proj1 (closure_env_invariant _ Hafter)).
Qed.

Theorem example_group_well_formed : example_group_well_formed_statement.
Proof. vm_compute. repeat split; reflexivity. Qed.
