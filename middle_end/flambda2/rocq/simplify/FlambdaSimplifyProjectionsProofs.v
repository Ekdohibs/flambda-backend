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


(* Proofs of the statements of FlambdaSimplifyProjectionsSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaHeapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaProvers
  FlambdaProversSpec FlambdaProversProofs FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyPrimitivesProofs FlambdaSimplifyProjectionsSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* A [Simple] that may occur in terms                                       *)
(* ======================================================================== *)

Theorem simple_in_terms_agrees : simple_in_terms_agrees_statement.
Proof.
  intros tenv valuation vars s Hagrees Hterms. symmetry.
  exact (agrees_eval_simple tenv valuation vars s Hagrees Hterms).
Qed.

(* A name mode that is at least the normal one is the normal one. *)
Lemma at_least_normal (name_mode : Name_mode.t) :
  name_mode_at_least name_mode Name_mode.Normal ->
  name_mode = Name_mode.Normal.
Proof.
  intros [c [Hcompare Hc]].
  destruct name_mode; [ reflexivity | | ]; simpl in Hcompare;
    injection Hcompare as <-; lia.
Qed.

Theorem canonical_simple_in_terms : canonical_simple_in_terms_statement.
Proof.
  intros tenv s canonical. unfold Typing_env.get_canonical_simple_exn.
  destruct (Typing_env.binding_time_and_mode_of_simple tenv s)
    as [binding_time_and_mode | ] eqn:Hmode_s; [ | discriminate ].
  simpl. intros [= Hcanonical] y ->.
  apply in_scope_in_terms_lookup.
  destruct (canonical_at_mode_name_mode _ _ _ _ _ _ _ _ _ Hcanonical)
    as [[<- Hmode] | Hmode]; apply at_least_normal in Hmode.
  - (* The [Simple] asked about: the environment read its name mode in
       scope. *)
    simpl in Hmode_s.
    unfold Typing_env.binding_time_and_mode,
      Typing_env.find_with_binding_time_and_mode,
      Typing_env.find_with_binding_time_and_mode' in Hmode_s.
    destruct (Keyed.lookup name_eqb (Name_var y)
                (Typing_env.names_to_types tenv))
      as [[ty binding_time_and_mode'] | ]; [ | discriminate ].
    exists ty, binding_time_and_mode'. split; [ reflexivity | ].
    simpl in Hmode_s.
    destruct (Typing_env.is_bottom tenv); simpl in Hmode_s;
      injection Hmode_s as <-; exact Hmode.
  - (* Another: its name mode in scope is in the environment. *)
    unfold Aliases.compute_name_mode, Aliases.binding_time_and_name_mode
      in Hmode.
    destruct (Keyed.lookup name_eqb (Name_var y)
                (Typing_env.names_to_types tenv))
      as [[ty binding_time_and_mode'] | ].
    + exists ty, binding_time_and_mode'. split; [ reflexivity | exact Hmode ].
    + exfalso. revert Hmode.
      unfold Binding_time.With_name_mode.scoped_name_mode. simpl.
      destruct (Typing_env.min_binding_time tenv
                <=? Binding_time.imported_variables)%Z;
        simpl; discriminate.
Qed.

(* What the provers call: the canonical element of the alias that a
   type is. *)
Lemma canonical_simple_of_alias_in_terms (tenv : Typing_env.t)
    (t : flambda_type) (s : simple) :
  Provers.canonical_simple_of_alias tenv Name_mode.normal t = Some (Some s) ->
  simple_in_terms tenv s.
Proof.
  unfold Provers.canonical_simple_of_alias.
  destruct (get_alias_opt t) as [alias | ]; [ | discriminate ].
  apply canonical_simple_in_terms.
Qed.

(* A meet shortcut on the head of a type of values answers what the
   function on heads proves. *)
Lemma gen_value_to_meet_known {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (t : flambda_type) (x : A) :
  Provers.gen_value_to_meet prove_gen tenv t
  = Some (Meet_shortcut.Known_result x) ->
  ∃ head : head_of_kind_value_non_null,
    prove_gen tenv head = Some (Generic_proof.Proved x).
Proof.
  unfold Provers.gen_value_to_meet.
  destruct (Provers.expand_head tenv t) as [d | ]; [ | discriminate ].
  simpl.
  destruct d as [[ | [[ | head | ] is_null] | ] | oub | k oub | oub | oub];
    try discriminate.
  destruct (prove_gen tenv head) as [[x' | | ] | ] eqn:Hhead;
    try discriminate.
  simpl. intros [= <-]. exists head. exact Hhead.
Qed.

(* The cases of a function on heads that answers a canonical element. *)
Ltac answer_cases :=
  repeat match goal with
         | |- context [match ?e with _ => _ end] =>
           destruct e eqn:?; try discriminate
         end.

Theorem provers_answer_in_terms : provers_answer_in_terms_statement.
Proof.
  split; [ | split ].
  - intros field_kind field_index tenv t s Hanswer.
    destruct (gen_value_to_meet_known _ tenv t s Hanswer) as [head Hhead].
    revert Hhead. unfold Provers.meet_block_field_simple_value.
    answer_cases. intros [= <-].
    eapply canonical_simple_of_alias_in_terms. eassumption.
  - intros move_to tenv t s Hanswer.
    destruct (gen_value_to_meet_known _ tenv t s Hanswer) as [head Hhead].
    revert Hhead. unfold Provers.meet_project_function_slot_simple_value.
    answer_cases. intros [= <-].
    eapply canonical_simple_of_alias_in_terms. eassumption.
  - intros slot value_slot_kind tenv t s Hanswer.
    destruct (gen_value_to_meet_known _ tenv t s Hanswer) as [head Hhead].
    revert Hhead. unfold Provers.meet_project_value_slot_simple_value.
    answer_cases. intros [= <-].
    eapply canonical_simple_of_alias_in_terms. eassumption.
Qed.

(* ======================================================================== *)
(* The conclusion                                                           *)
(* ======================================================================== *)

Theorem let_simple : let_simple_statement.
Proof.
  intros tenv x declared d s body Hknown vars conts statics hp rd Hsat.
  apply let_refines_step_forward.
  - intros v hp' rd' Hd.
    destruct (Hknown statics hp hp' rd rd' vars v Hsat Hd) as (Hs & -> & ->).
    simpl. rewrite Hs. reflexivity.
  - intros v hp' rd' _ _. apply refines_refl.
Qed.

(* The conclusion with the premise on the value alone
   ([eval_named]), as FlambdaSimplifyPrimitivesProofs.v has the two
   others: of a defining expression that reads no cell, and of a unary
   primitive on an argument whose type has no location. *)
Lemma let_simple_pure (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named) (s : simple)
    (body : expr) :
  named_reads_heap defining_expr = false ->
  (∀ (statics : static_env) (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars defining_expr = Some v ->
    eval_simple vars s = Some v) ->
  expr_refines_under tenv (Let x declared defining_expr body)
    (Let x declared (Simple s) body).
Proof.
  intros Hpure Hknown. apply let_simple.
  intros statics hp hp' rd rd' vars v Hsat Hexec.
  destruct (exec_named_pure statics hp hp' rd rd' vars defining_expr v Hpure
              Hexec) as (Heval & -> & ->).
  split; [ exact (Hknown statics vars v Hsat Heval) | split; reflexivity ].
Qed.

Lemma let_simple_unary (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (p : unary_primitive) (arg : simple)
    (ty : flambda_type) (s : simple) (body : expr) :
  argument_type tenv arg ty ->
  no_location tenv ty ->
  (∀ (statics : static_env) (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars (Prim (Unary p) [arg]) = Some v ->
    eval_simple vars s = Some v) ->
  expr_refines_under tenv (Let x declared (Prim (Unary p) [arg]) body)
    (Let x declared (Simple s) body).
Proof.
  intros Hty Hno Hknown. apply let_simple.
  intros statics hp hp' rd rd' vars v Hsat Hexec.
  destruct (unary_prim_exec_typed tenv statics hp hp' rd rd' vars p arg ty v
              Hsat Hty Hno Hexec) as (Hnamed & -> & ->).
  split; [ exact (Hknown statics vars v Hsat Hnamed) | split; reflexivity ].
Qed.

(* ======================================================================== *)
(* A load from a block                                                      *)
(* ======================================================================== *)

Theorem let_block_load : let_block_load_statement.
Proof.
  intros tenv x declared kind mut field arg ty body _ Hty load.
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars load = Some r ->
      match
        Provers.meet_block_field_simple tenv Name_mode.normal Value ty
          (Z.to_nat field)
      with
      | Some (Meet_shortcut.Known_result s) => eval_simple vars s = Some r
      | Some Meet_shortcut.Invalid => False
      | Some Meet_shortcut.Need_meet | None => True
      end).
  { intros statics vars r Hsat Hnamed.
    destruct (unary_prim_value statics vars _ arg r Hnamed) as [v [Hv Hprim]].
    destruct (argument_type_satisfied tenv statics vars arg ty v Hsat Hty Hv)
      as [valuation [Hsatisfied [Hagrees Htyped]]].
    destruct (block_load_defined statics kind mut field v r Hprim)
      as [_ [Hdomain [tag [fields [Hblock Hnth]]]]].
    pose proof (meet_block_field_simple_sound Name_mode.normal Value
                  (Z.to_nat field) statics valuation tenv ty v Hsatisfied I
                  Htyped Hdomain) as Hanswer.
    pose proof (proj1 provers_answer_in_terms Value (Z.to_nat field) tenv ty)
      as Hterms.
    cbv beta in Hanswer, Hterms.
    destruct (Provers.meet_block_field_simple tenv Name_mode.normal Value ty
                (Z.to_nat field)) as [[s | | ] | ];
      try exact I; [ | exact Hanswer ].
    destruct Hanswer as [tag' [fields' [field' [Hblock' [Hnth' Hs]]]]].
    rewrite Hblock in Hblock'. injection Hblock' as <- <-.
    rewrite Hnth in Hnth'. injection Hnth' as <-.
    rewrite (simple_in_terms_agrees tenv valuation vars s Hagrees
               (Hterms s eq_refl)).
    exact Hs. }
  destruct (Provers.meet_block_field_simple tenv Name_mode.normal Value ty
              (Z.to_nat field)) as [[s | | ] | ] eqn:Hanswer;
    try exact I.
  - apply (let_simple_unary tenv x declared _ arg ty s body Hty);
      [ exact (meet_answer_no_location _ tenv ty _ Hanswer
                 ltac:(discriminate))
      | exact Hsound ].
  - apply (let_no_value_unary tenv x declared _ arg ty body Hty);
      [ exact (meet_answer_no_location _ tenv ty _ Hanswer
                 ltac:(discriminate))
      | exact Hsound ].
Qed.

(* ======================================================================== *)
(* The projections from a closure                                           *)
(* ======================================================================== *)

Theorem let_project_function_slot : let_project_function_slot_statement.
Proof.
  intros tenv x declared move_from move_to closure ty body Hty
    projection.
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars projection = Some r ->
      match
        Provers.meet_project_function_slot_simple tenv Name_mode.normal ty
          move_to
      with
      | Some (Meet_shortcut.Known_result s) => eval_simple vars s = Some r
      | Some Meet_shortcut.Invalid => False
      | Some Meet_shortcut.Need_meet | None => True
      end).
  { intros statics vars r Hsat Hnamed. cbn [projection eval_named] in Hnamed.
    destruct (eval_simple vars closure) as [v | ] eqn:Hv; [ | discriminate ].
    simpl in Hnamed.
    destruct (argument_type_satisfied tenv statics vars closure ty v Hsat Hty
                Hv) as [valuation [Hsatisfied [Hagrees Htyped]]].
    pose proof (meet_project_function_slot_simple_sound
                  Name_mode.normal move_to statics valuation tenv ty v
                  Hsatisfied I Htyped
                  (project_function_slot_in_domain statics move_from move_to
                     v r Hnamed)) as Hanswer.
    pose proof (proj1 (proj2 provers_answer_in_terms) move_to tenv ty)
      as Hterms.
    cbv beta in Hanswer, Hterms.
    destruct (Provers.meet_project_function_slot_simple tenv Name_mode.normal
                ty move_to) as [[s | | ] | ];
      try exact I; [ | exact Hanswer ].
    rewrite (simple_in_terms_agrees tenv valuation vars s Hagrees
               (Hterms s eq_refl)).
    exact (Hanswer move_from r Hnamed). }
  destruct (Provers.meet_project_function_slot_simple tenv Name_mode.normal
              ty move_to) as [[s | | ] | ];
    try exact I.
  - apply let_simple_pure; [ reflexivity | exact Hsound ].
  - apply let_no_value_pure; [ reflexivity | exact Hsound ].
Qed.

Theorem let_project_value_slot : let_project_value_slot_statement.
Proof.
  intros tenv x declared project_from slot value_slot_kind closure
    ty body Hty projection Hkind.
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars projection = Some r ->
      match
        Provers.meet_project_value_slot_simple tenv Name_mode.normal ty slot
          value_slot_kind
      with
      | Some (Meet_shortcut.Known_result s) => eval_simple vars s = Some r
      | Some Meet_shortcut.Invalid => False
      | Some Meet_shortcut.Need_meet | None => True
      end).
  { intros statics vars r Hsat Hnamed.
    pose proof (Hkind statics vars r Hsat Hnamed) as Hr.
    cbn [projection eval_named] in Hnamed.
    destruct (eval_simple vars closure) as [v | ] eqn:Hv; [ | discriminate ].
    simpl in Hnamed.
    destruct (argument_type_satisfied tenv statics vars closure ty v Hsat Hty
                Hv) as [valuation [Hsatisfied [Hagrees Htyped]]].
    pose proof (meet_project_value_slot_simple_sound Name_mode.normal
                  slot value_slot_kind statics valuation tenv ty v Hsatisfied
                  I Htyped
                  (project_value_slot_in_domain statics project_from slot
                     value_slot_kind v r Hnamed Hr)) as Hanswer.
    pose proof (proj2 (proj2 provers_answer_in_terms) slot value_slot_kind
                  tenv ty) as Hterms.
    cbv beta in Hanswer, Hterms.
    destruct (Provers.meet_project_value_slot_simple tenv Name_mode.normal ty
                slot value_slot_kind) as [[s | | ] | ];
      try exact I; [ | exact Hanswer ].
    rewrite (simple_in_terms_agrees tenv valuation vars s Hagrees
               (Hterms s eq_refl)).
    exact (Hanswer project_from r Hnamed). }
  destruct (Provers.meet_project_value_slot_simple tenv Name_mode.normal ty
              slot value_slot_kind) as [[s | | ] | ];
    try exact I.
  - apply let_simple_pure; [ reflexivity | exact Hsound ].
  - apply let_no_value_pure; [ reflexivity | exact Hsound ].
Qed.

(* ======================================================================== *)
(* A block that is made                                                     *)
(* ======================================================================== *)

(* The type that the environment answers for a [Simple] of a term is
   the alias type of the canonical element answered. *)
Lemma type_simple_in_term_alias (tenv : Typing_env.t)
    (min_name_mode : option Name_mode.t) (s canonical : simple)
    (ty : flambda_type) :
  Typing_env.type_simple_in_term_exn tenv min_name_mode s
  = Some (Some (ty, canonical)) ->
  ∃ kind : flambda_kind, ty = alias_type_of kind canonical.
Proof.
  unfold Typing_env.type_simple_in_term_exn.
  destruct (match s with
            | Const const =>
              Some (type_for_const const, Binding_time.With_name_mode.consts)
            | Var x =>
              Typing_env.find_with_binding_time_and_mode tenv (Name_var x) None
            | Symbol sym =>
              Typing_env.find_with_binding_time_and_mode tenv
                (Name_symbol sym) None
            end) as [[ty' binding_time_and_mode] | ]; [ | discriminate ].
  destruct (Aliases.get_canonical_element_exn _ _ _ _ _ _ _) as [alias | ];
    [ | discriminate ].
  intros [= <- <-]. eexists. reflexivity.
Qed.

Theorem argument_type_in_machine : argument_type_in_machine_statement.
Proof.
  intros tenv statics vars arg ty v Hsat [Hmem [canonical Htype]] Heval.
  pose proof (type_simple_in_term_canonical tenv statics vars arg canonical
                ty Hsat Hmem Htype) as Hcanonical.
  destruct Hsat as [_ [valuation [Hsatisfied Hagrees]]].
  assert (Htyped : type_satisfied statics valuation v ty).
  { apply (proj2 (type_simple_in_term_satisfied _ statics valuation tenv
                    (Some Name_mode.normal) arg canonical ty Hsatisfied
                    Htype)).
    rewrite (agrees_eval_simple tenv valuation vars arg Hagrees);
      [ exact Heval | ].
    intros x ->. exact Hmem. }
  destruct (type_simple_in_term_alias _ _ _ _ _ Htype) as [kind ->].
  apply alias_satisfied in Htyped. apply alias_satisfied.
  split; [ rewrite Hcanonical; exact Heval | exact (proj2 Htyped) ].
Qed.

Lemma free_variables_alias_type (kind : flambda_kind) (s : simple)
    (y : variable) :
  y ∈ free_variables (alias_type_of kind s) -> s = Var y.
Proof.
  destruct kind as [ | [] | | ]; destruct s as [x | sym | c]; cbn;
    intros Hin; try contradiction; destruct Hin as [-> | []]; reflexivity.
Qed.

Theorem argument_type_in_scope : argument_type_in_scope_statement.
Proof.
  intros tenv arg ty y [Hmem [canonical Htype]] Hfree.
  destruct (type_simple_in_term_alias _ _ _ _ _ Htype) as [kind ->].
  apply free_variables_alias_type in Hfree. subst canonical.
  revert Htype. unfold Typing_env.type_simple_in_term_exn.
  destruct (match arg with
            | Const const =>
              Some (type_for_const const, Binding_time.With_name_mode.consts)
            | Var x =>
              Typing_env.find_with_binding_time_and_mode tenv (Name_var x) None
            | Symbol sym =>
              Typing_env.find_with_binding_time_and_mode tenv
                (Name_symbol sym) None
            end) as [[ty' binding_time_and_mode] | ]; [ | discriminate ].
  destruct (Aliases.get_canonical_element_exn _ _ _ _ _ _ _) as [alias | ]
    eqn:Halias; [ | discriminate ].
  intros [= _ ->].
  apply get_canonical_element_exn_normal in Halias.
  destruct Halias as [-> | [data [binding_time_and_mode' [Hlookup Hmode]]]].
  - exact Hmem.
  - apply in_scope_in_terms_lookup.
    exists data, binding_time_and_mode'. split; assumption.
Qed.

Theorem make_block_typed : make_block_typed_statement.
Proof.
  intros tenv statics vars tag args tys v Hsat Htys Htag Hnamed.
  cbn [eval_named] in Hnamed.
  destruct (eval_simples vars args) as [vs | ] eqn:Hvs; [ | discriminate ].
  simpl in Hnamed. rewrite eval_prim_variadic in Hnamed. simpl in Hnamed.
  destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs)
    eqn:Hkinds; [ | discriminate ].
  destruct (forallb (bound_if_symbol statics) vs); [ | discriminate ].
  injection Hnamed as <-.
  apply immutable_block_satisfied. exists vs.
  split; [ reflexivity | ]. split.
  - split; [ exact Htag | ]. simpl. clear Hvs.
    induction vs as [ | v vs IH]; [ reflexivity | ].
    simpl in Hkinds. apply andb_true_iff in Hkinds.
    destruct Hkinds as [Hv Hrest]. simpl. f_equal; [ | exact (IH Hrest) ].
    destruct (kind_of_value v); try discriminate. reflexivity.
  - pose proof (eval_simples_values vars args vs Hvs) as Hvalues.
    clear Hvs Hkinds. revert vs Hvalues.
    induction Htys as [ | arg ty args tys Hty Htys IH]; intros vs Hvalues;
      inversion Hvalues as [ | arg' v args' vs' Hv Hrest]; subst;
      [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact (IH vs' Hrest) ].
    exact (argument_type_in_machine tenv statics vars arg ty v Hsat Hty Hv).
Qed.

(* The variables of the type of a block that is made are those of the
   types of its fields. *)
Lemma free_variables_make_block_type (tag : Z) (tys : list flambda_type)
    (y : variable) :
  y ∈ free_variables (make_block_type tag tys) ->
  ∃ ty : flambda_type, ty ∈ tys ∧ y ∈ free_variables ty.
Proof.
  unfold make_block_type, immutable_block. cbn. rewrite !app_nil_r.
  intros Hin. apply in_flat_map in Hin. exact Hin.
Qed.

Theorem let_make_block : let_make_block_statement.
Proof.
  intros tenv x declared tag args tys body body' Hwf Hfresh Htys Htag
    make_block Hbody.
  apply (let_typed_refines_under_pure tenv x declared make_block body body'
           (make_block_type tag tys) eq_refl Hwf Hfresh).
  - intros y Hy.
    destruct (free_variables_make_block_type tag tys y Hy)
      as [ty [Hin Hfree]].
    clear Hbody Hy. induction Htys as [ | arg ty' args tys Hty Htys IH];
      [ contradiction | ].
    destruct Hin as [<- | Hin];
      [ exact (argument_type_in_scope tenv arg ty' y Hty Hfree)
      | exact (IH Hin) ].
  - intros statics vars v Hsat Hnamed.
    exact (make_block_typed tenv statics vars tag args tys v
             (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
             Htys Htag Hnamed).
  - exact Hbody.
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem example_first_of_pair : example_first_of_pair_statement.
Proof.
  intros k pair.
  apply (let_make_block example_pair_env example_p Value 0
           [Var example_x; Var example_y]
           [alias_type_of Value (Var example_x);
            alias_type_of Value (Var example_y)]);
    [ vm_compute; reflexivity | vm_compute; reflexivity
    | repeat apply Forall2_cons;
      [ argument_computed | argument_computed | apply Forall2_nil ]
    | reflexivity | ].
  eapply expr_refines_under_trans.
  - (* The load is the [Simple] that the prover answers. *)
    match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_block_load tenv example_q Value
                    (Block_access_values (Some 0) (Some 2) Any_value)
                    Immutable 0 (Var example_p)
                    (alias_type_of Value (Var example_p)) body) as Hload
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
