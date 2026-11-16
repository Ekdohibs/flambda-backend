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


(* The proofs of the statements of FlambdaJoinUsesSpec.v. This file
   needs no review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec
  FlambdaPrimitiveProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeWellFormedSpec FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs FlambdaEnvAliasesInvariantSpec
  FlambdaEnvAliasesInvariantProofs FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinTypesExamplesSpec FlambdaMeetEnv FlambdaMeetSpec
  FlambdaMeetStepsSpec FlambdaMeetStepsProofs FlambdaMeetEnvSpec
  FlambdaMeetEnvProofs FlambdaMeetScopesProofs FlambdaJoinMeetFitsSpec
  FlambdaJoinInTermsSpec FlambdaJoinUsesSpec.
Import ListNotations.

Local Open Scope Z_scope.

Theorem alias_with_equation_is_arms_alias
  : alias_with_equation_is_arms_alias_statement.
Proof. intros kind tenv x simple. reflexivity. Qed.

(* ======================================================================== *)
(* What a step keeps                                                        *)
(* ======================================================================== *)

(* The second environment is the first with equations added: it
   extends it, knows no other variable, and is bottom if the first
   is. *)
Definition with_equations (tenv tenv' : Typing_env.t) : Prop :=
  tenv_extends tenv tenv'
  ∧ no_new_variables tenv tenv'
  ∧ Typing_env.is_bottom tenv' = Typing_env.is_bottom tenv.

Lemma with_equations_refl (tenv : Typing_env.t) : with_equations tenv tenv.
Proof.
  split; [ apply tenv_extends_refl | ]. split; [ | reflexivity ].
  intros x Hknown. exact Hknown.
Qed.

Lemma with_equations_trans (tenv1 tenv2 tenv3 : Typing_env.t) :
  with_equations tenv1 tenv2 -> with_equations tenv2 tenv3 ->
  with_equations tenv1 tenv3.
Proof.
  intros (Hextends12 & Hnew12 & Hbottom12) (Hextends23 & Hnew23 & Hbottom23).
  split; [ exact (tenv_extends_trans _ _ _ Hextends12 Hextends23) | ].
  split; [ | congruence ].
  intros x Hknown. exact (Hnew12 x (Hnew23 x Hknown)).
Qed.

(* A name is given a type. *)
Lemma replace_equation_with_equations (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) :
  typing_env_invariant tenv ->
  simple_of_kind tenv (simple_of_name n) (kind ty) ->
  with_equations tenv (Typing_env.replace_equation tenv n ty).
Proof.
  intros Hinvariant Hkind.
  split; [ exact (replace_equation_tenv_extends tenv n ty Hinvariant Hkind) | ].
  split; [ exact (replace_equation_no_new tenv n ty) | ].
  assert (Hframe := replace_equation_frame tenv n ty). cbv zeta in Hframe.
  exact (proj1 (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 Hframe)))))))).
Qed.

(* A variable is made equal to a [Simple]. *)
Lemma alias_with_equation_keeps (kind : flambda_kind)
    (tenv tenv' : Typing_env.t) (x : variable) (simple : simple) :
  typing_env_fits tenv ->
  is_canonical (Typing_env.aliases tenv) (Var x) ->
  is_canonical (Typing_env.aliases tenv) simple ->
  Var x <> simple ->
  simple_of_kind tenv (Var x) kind ->
  simple_of_kind tenv simple kind ->
  alias_with_equation kind tenv x simple = Some tenv' ->
  with_equations tenv tenv' ∧ typing_env_fits tenv'.
Proof.
  intros Hfits Hc1 Hc2 Hdifferent Hkind1 Hkind2 Hmade.
  assert (Hinvariant := proj1 Hfits).
  unfold alias_with_equation in Hmade.
  destruct (Typing_env.add_alias tenv (Var x) simple)
    as [ [ | result | ] | ] eqn:Halias; try discriminate Hmade.
  cbn in Hmade. injection Hmade as <-.
  assert (Hextends1 := add_alias_tenv_extends tenv _ _ result Halias).
  assert (Hfits1 :=
    add_alias_fits tenv _ _ result Halias kind Hfits Hc1 Hc2
      (simple_of_kind_mem_simple tenv _ kind Hinvariant Hkind1)
      (simple_of_kind_mem_simple tenv _ kind Hinvariant Hkind2)
      Hkind1 Hkind2).
  destruct (add_alias_demoted tenv _ _ result Halias Hinvariant Hc1 Hc2
              Hdifferent) as [Hwhich Hnot].
  assert (Hframe := add_alias_frame tenv _ _ result Halias).
  cbv zeta in Hframe.
  destruct Hframe as (_ & _ & _ & _ & _ & _ & _ & Hbottom1).
  set (tenv1 := Typing_env.add_alias_result_t result) in *.
  set (demoted := Typing_env.demoted_name result) in *.
  set (canonical := Typing_env.canonical_element result) in *.
  set (alias := alias_type_of kind canonical).
  assert (Hinvariant1 := proj1 Hfits1).
  assert (Hkept :
    ∀ s : FlambdaModel.simple,
      simple_of_kind tenv s kind -> simple_of_kind tenv1 s kind).
  { intros s.
    exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends1)))) s kind). }
  assert (Hkind_demoted : simple_of_kind tenv1 (simple_of_name demoted) kind).
  { apply Hkept. destruct Hwhich as [ [_ ->] | [_ ->] ]; assumption. }
  assert (Hkind_canonical : simple_of_kind tenv1 canonical kind).
  { apply Hkept. destruct Hwhich as [ [-> _] | [-> _] ]; assumption. }
  assert (Hkind_alias :
    simple_of_kind tenv1 (simple_of_name demoted)
      (FlambdaTypeGrammar.kind alias)).
  { unfold alias. rewrite kind_alias_type_of. exact Hkind_demoted. }
  split.
  - apply (with_equations_trans tenv tenv1).
    + split; [ exact Hextends1 | ].
      split; [ exact (add_alias_no_new tenv _ _ result Halias) | ].
      exact Hbottom1.
    + exact (replace_equation_with_equations tenv1 demoted alias Hinvariant1
               Hkind_alias).
  - apply (replace_equation_fits tenv1 demoted alias Hinvariant1 Hkind_alias
             Hfits1).
    + apply type_fits_alias; [ | exact Hkind_canonical ].
      exact (simple_of_kind_mem_simple tenv1 canonical kind Hinvariant1
               Hkind_canonical).
    + intros s _. exact Hnot.
    + apply is_obviously_unknown_alias_type_of.
    + apply is_obviously_bottom_alias_type_of.
Qed.

(* A use made from the source fits, and is the source one scope
   further with equations added. *)
Lemma made_from_keeps (source use : Typing_env.t) :
  typing_env_fits source ->
  made_from source use ->
  typing_env_fits use
  ∧ with_equations (Typing_env.increment_scope source) use.
Proof.
  intros Hsource Hmade.
  induction Hmade
    as [ | tenv tenv' kind x simple _ IH Hc1 Hc2 Hdifferent Hkind1 Hkind2
           Halias
       | tenv x ty _ IH Hkind Hty Halias Hunknown Hbottom ].
  - split; [ exact (increment_scope_fits source Hsource) | ].
    apply with_equations_refl.
  - destruct IH as [Hfits Hwith].
    destruct (alias_with_equation_keeps kind tenv tenv' x simple Hfits Hc1
                Hc2 Hdifferent Hkind1 Hkind2 Halias) as [Hstep Hfits'].
    split; [ exact Hfits' | ].
    exact (with_equations_trans _ _ _ Hwith Hstep).
  - destruct IH as [Hfits Hwith].
    assert (Hinvariant := proj1 Hfits).
    split.
    + apply (replace_equation_fits tenv (Name_var x) ty Hinvariant Hkind
               Hfits Hty); try assumption.
      intros alias Hsome. rewrite Halias in Hsome. discriminate Hsome.
    + apply (with_equations_trans _ tenv); [ exact Hwith | ].
      exact (replace_equation_with_equations tenv (Name_var x) ty Hinvariant
               Hkind).
Qed.

(* ======================================================================== *)
(* The premises on the inputs                                               *)
(* ======================================================================== *)

(* The level cut from an environment that is another one scope
   further with equations added, after the scope of that other. *)
Lemma cut_with_equations (source use : Typing_env.t) :
  tenv_extends (Typing_env.increment_scope source) use ->
  Typing_env.cut use (Typing_env.current_scope source)
  = Some (Typing_env.One_level.level (Typing_env.current_level use)).
Proof.
  intros (Hscope & Hprev & _).
  unfold Typing_env.cut. rewrite Hscope, Hprev.
  unfold Typing_env.increment_scope, Typing_env.current_scope.
  cbn [Typing_env.current_level Typing_env.prev_levels
         Typing_env.One_level.scope Typing_env.cut_loop].
  unfold Scope.next.
  destruct (Z.leb_spec
              (Typing_env.One_level.scope (Typing_env.current_level source)
               + 1)
              (Typing_env.One_level.scope (Typing_env.current_level source)))
    as [Hle | _]; [ lia | ].
  rewrite Z.ltb_irrefl. reflexivity.
Qed.

(* A variable that may occur in terms in an environment may occur in
   terms in one that extends it. *)
Lemma tenv_extends_in_scope_in_terms (tenv tenv' : Typing_env.t)
    (x : variable) :
  tenv_extends tenv tenv' ->
  in_scope_in_terms tenv x -> in_scope_in_terms tenv' x.
Proof.
  intros (_ & _ & Hmin & _ & _ & Hkeeps).
  unfold in_scope_in_terms, Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv))
    as [ [ty binding_time_and_mode] | ] eqn:Hlookup; [ | discriminate ].
  destruct (Hkeeps _ _ _ Hlookup) as (ty' & Hlookup' & _).
  rewrite Hlookup', Hmin. intros Hscope. exact Hscope.
Qed.

Theorem uses_made_from_source : uses_made_from_source_statement.
Proof.
  intros A avoid source uses Hsource Havoid Hmade inputs.
  assert (Huse :
    ∀ (index : Index.t) (use : Typing_env.t),
      use_at inputs index use ->
      typing_env_fits use
      ∧ with_equations (Typing_env.increment_scope source) use).
  { intros index use [id Hnth]. apply nth_error_In in Hnth.
    exact (made_from_keeps source use Hsource (Hmade id use Hnth)). }
  assert (Hknows :
    ∀ (index : Index.t) (use : Typing_env.t) (x : variable),
      use_at inputs index use -> (knows use x <-> knows source x)).
  { intros index use x Hat.
    destruct (Huse index use Hat) as (_ & Hextends & Hnew & _). split.
    - intros Hknown. exact (Hnew x Hknown).
    - intros Hknown.
      exact (tenv_extends_mem _ _ (Name_var x) Hextends Hknown). }
  split.
  { (* [join_inputs_well_formed] *)
    split.
    { split; [ exact (proj1 Hsource) | ].
      intros index use Hat. exact (proj1 (proj1 (Huse index use Hat))). }
    split.
    { split; [ exact (proj2 (proj2 (proj2 (proj2 Hsource)))) | ].
      intros index use Hat.
      exact (proj2 (proj2 (proj2 (proj2 (proj1 (Huse index use Hat)))))). }
    split.
    { intros index use Hat _.
      destruct (Huse index use Hat) as (_ & Hextends & _).
      eexists. exact (cut_with_equations source use Hextends). }
    split.
    { intros index use sym Hat Hmem.
      destruct (Huse index use Hat) as (_ & (_ & _ & _ & Hsymbols & _) & _).
      unfold Typing_env.mem in *. rewrite Hsymbols in Hmem. exact Hmem. }
    split; [ exact Havoid | ].
    intros index use x Hat Hknown.
    exact (Havoid x (proj1 (Hknows index use x Hat) Hknown)). }
  split.
  { intros index use Hat.
    exact (proj1 (proj2 (proj2 (proj1 (Huse index use Hat))))). }
  split.
  { intros index use n kind ty Hat _ Hfind.
    destruct (Huse index use Hat) as (_ & Hextends & _).
    assert (Hkind :
      simple_of_kind use (simple_of_name n) kind).
    { apply (proj1 (proj2 (proj2 (proj2 (proj2 Hextends))))).
      apply increment_scope_simple_of_kind.
      destruct n; exists ty; exact Hfind. }
    destruct n; exact Hkind. }
  split.
  { intros index use Hat _.
    destruct (Huse index use Hat) as ((_ & Hwf & _ & Hequations & _) & _).
    split; assumption. }
  split; [ exact Hknows | ].
  split.
  { intros index use x Hat Hscope.
    destruct (Huse index use Hat) as (_ & Hextends & _).
    exact (tenv_extends_in_scope_in_terms _ _ x Hextends Hscope). }
  intros id use Hin.
  exact (proj2 (proj2 (proj2 (made_from_keeps source use Hsource
                                (Hmade id use Hin))))).
Qed.

(* ======================================================================== *)
(* The examples                                                             *)
(* ======================================================================== *)

(* [simple_of_kind], as a test. *)
Definition of_kind_b (tenv : Typing_env.t) (s : simple) (kind : flambda_kind)
    : bool :=
  match s with
  | Const c => kind_eqb (FlambdaTypeGrammar.kind (type_for_const c)) kind
  | Var x => is_some (Typing_env.find tenv (Name_var x) (Some kind))
  | Symbol sym => is_some (Typing_env.find tenv (Name_symbol sym) (Some kind))
  end.

Lemma of_kind_b_sound (tenv : Typing_env.t) (s : simple)
    (kind : flambda_kind) :
  of_kind_b tenv s kind = true -> simple_of_kind tenv s kind.
Proof.
  destruct s as [ x | sym | c ]; cbn [of_kind_b simple_of_kind].
  - destruct (Typing_env.find tenv (Name_var x) (Some kind)) as [ ty | ];
      [ intros _; exists ty; reflexivity | discriminate ].
  - destruct (Typing_env.find tenv (Name_symbol sym) (Some kind)) as [ ty | ];
      [ intros _; exists ty; reflexivity | discriminate ].
  - intros Hkind. apply kind_eqb_eq. exact Hkind.
Qed.

(* A meet may be made in the environment that knows nothing. *)
Lemma create_fits : typing_env_fits Typing_env.create.
Proof.
  split; [ exact create_invariant | ].
  split; [ exact create_types_well_formed | ].
  split; [ exact create_kinded | ].
  split; [ exact create_equations_kinded | ].
  intros _ n ty binding_time_and_mode Hlookup. discriminate Hlookup.
Qed.

(* A variable is defined. *)
Lemma define_fits (tenv : Typing_env.t) (x : variable) (kind : flambda_kind) :
  typing_env_fits tenv ->
  Typing_env.mem None tenv (Name_var x) = false ->
  typing_env_fits (Arms.define tenv x kind).
Proof.
  intros Hfits Hfresh.
  exact (proj2 (add_variable_definition_keeps (Meet_env.create tenv) x kind
                  Name_mode.normal Hfits Hfresh)).
Qed.

(* A variable is defined as a block whose field is another. *)
Lemma define_block_fits (tenv : Typing_env.t) (x field : variable) :
  typing_env_fits (Arms.define tenv x Value) ->
  of_kind_b (Arms.define tenv x Value) (Var x) Value = true ->
  of_kind_b (Arms.define tenv x Value) (Var field) Value = true ->
  typing_env_fits (Arms.define_block tenv x field).
Proof.
  intros Hfits Hkind_x Hkind_field.
  apply of_kind_b_sound in Hkind_x, Hkind_field.
  assert (Hinvariant := proj1 Hfits).
  unfold Arms.define_block.
  apply (replace_equation_fits (Arms.define tenv x Value) (Name_var x)
           (example_join_block 0 [alias_type_of Value (Var field)])
           Hinvariant Hkind_x Hfits).
  - split; [ reflexivity | ]. split.
    { intros y Hin. cbn in Hin. destruct Hin as [ <- | [] ].
      exact (simple_of_kind_mem_simple _ (Var field) Value Hinvariant
               Hkind_field). }
    split.
    + intros s kind Hin. cbn in Hin.
      destruct Hin as [ [= <- <-] | [] ]. exact Hkind_field.
    + intros n kind Hin. cbn in Hin. destruct Hin.
  - intros alias Hsome. discriminate Hsome.
  - reflexivity.
  - reflexivity.
Qed.

Lemma arms_source_fits : typing_env_fits Arms.source_env.
Proof.
  unfold Arms.source_env, Arms.at_switch.
  apply (define_fits _ Arms.q Value); [ | vm_compute; reflexivity ].
  apply (define_fits _ Arms.p Value); [ | vm_compute; reflexivity ].
  apply (define_block_fits _ Arms.z2 Arms.a2);
    [ | vm_compute; reflexivity | vm_compute; reflexivity ].
  apply (define_fits _ Arms.z2 Value); [ | vm_compute; reflexivity ].
  apply (define_block_fits _ Arms.z1 Arms.a1);
    [ | vm_compute; reflexivity | vm_compute; reflexivity ].
  apply (define_fits _ Arms.z1 Value); [ | vm_compute; reflexivity ].
  apply (define_block_fits _ Arms.y2 Arms.a2);
    [ | vm_compute; reflexivity | vm_compute; reflexivity ].
  apply (define_fits _ Arms.y2 Value); [ | vm_compute; reflexivity ].
  apply (define_block_fits _ Arms.y1 Arms.a1);
    [ | vm_compute; reflexivity | vm_compute; reflexivity ].
  apply (define_fits _ Arms.y1 Value); [ | vm_compute; reflexivity ].
  apply (define_fits _ Arms.a2 Value); [ | vm_compute; reflexivity ].
  apply (define_fits _ Arms.a1 Value); [ | vm_compute; reflexivity ].
  apply (define_fits _ Arms.s (Naked_number Kind_naked_immediate));
    [ | vm_compute; reflexivity ].
  exact create_fits.
Qed.

(* [alias_with_equation], with the side conditions of [made_from]
   tested. *)
Definition checked_alias (kind : flambda_kind) (tenv : Typing_env.t)
    (x : variable) (simple : simple) : option Typing_env.t :=
  if is_canonical_b (Typing_env.aliases tenv) (Var x)
     && is_canonical_b (Typing_env.aliases tenv) simple
     && negb (simple_equal (Var x) simple)
     && of_kind_b tenv (Var x) kind
     && of_kind_b tenv simple kind
  then alias_with_equation kind tenv x simple
  else None.

Lemma checked_alias_made (source : Typing_env.t) (kind : flambda_kind)
    (tenv tenv' : Typing_env.t) (x : variable) (simple : simple) :
  made_from source tenv ->
  checked_alias kind tenv x simple = Some tenv' ->
  made_from source tenv'.
Proof.
  intros Hmade Hchecked. unfold checked_alias in Hchecked.
  destruct (is_canonical_b (Typing_env.aliases tenv) (Var x)) eqn:Hc1;
    [ | discriminate Hchecked ].
  destruct (is_canonical_b (Typing_env.aliases tenv) simple) eqn:Hc2;
    [ | discriminate Hchecked ].
  destruct (simple_equal (Var x) simple) eqn:Hsame;
    [ discriminate Hchecked | ].
  destruct (of_kind_b tenv (Var x) kind) eqn:Hkind1;
    [ | discriminate Hchecked ].
  destruct (of_kind_b tenv simple kind) eqn:Hkind2;
    [ | discriminate Hchecked ].
  cbn in Hchecked.
  apply (Made_equal source tenv tenv' kind x simple Hmade).
  - apply is_canonical_b_iff. exact Hc1.
  - apply is_canonical_b_iff. exact Hc2.
  - intros Heq. apply simple_equal_eq in Heq. rewrite Heq in Hsame.
    discriminate Hsame.
  - exact (of_kind_b_sound _ _ _ Hkind1).
  - exact (of_kind_b_sound _ _ _ Hkind2).
  - exact Hchecked.
Qed.

(* [Arms.use] with equations, with the side conditions tested. *)
Definition checked_use (discriminant : Z)
    (argument_of_p argument_of_q : simple) : option Typing_env.t :=
  let* tenv :=
    checked_alias (Naked_number Kind_naked_immediate)
      (Typing_env.increment_scope Arms.source_env) Arms.s
      (Const (Naked_immediate discriminant))
  in
  let* tenv := checked_alias Value tenv Arms.p argument_of_p in
  checked_alias Value tenv Arms.q argument_of_q.

Lemma checked_use_made (discriminant : Z)
    (argument_of_p argument_of_q : simple) (tenv : Typing_env.t) :
  checked_use discriminant argument_of_p argument_of_q = Some tenv ->
  made_from Arms.source_env tenv.
Proof.
  unfold checked_use. intros Hchecked.
  destruct (checked_alias (Naked_number Kind_naked_immediate)
              (Typing_env.increment_scope Arms.source_env) Arms.s
              (Const (Naked_immediate discriminant)))
    as [ tenv1 | ] eqn:H1; [ | discriminate Hchecked ].
  cbv beta iota in Hchecked.
  destruct (checked_alias Value tenv1 Arms.p argument_of_p)
    as [ tenv2 | ] eqn:H2; [ | discriminate Hchecked ].
  cbv beta iota in Hchecked.
  apply (checked_alias_made Arms.source_env Value tenv2 tenv Arms.q
           argument_of_q); [ | exact Hchecked ].
  apply (checked_alias_made Arms.source_env Value tenv1 tenv2 Arms.p
           argument_of_p); [ | exact H2 ].
  apply (checked_alias_made Arms.source_env
           (Naked_number Kind_naked_immediate)
           (Typing_env.increment_scope Arms.source_env) tenv1 Arms.s
           (Const (Naked_immediate discriminant))); [ | exact H1 ].
  apply Made_one_scope_further.
Qed.

(* A use of the examples is the one with the side conditions tested,
   which answers. *)
Lemma arms_use_made (discriminant : Z) (argument_of_p argument_of_q : simple) :
  Arms.use true discriminant argument_of_p argument_of_q
  = checked_use discriminant argument_of_p argument_of_q ->
  is_some (checked_use discriminant argument_of_p argument_of_q) = true ->
  ∃ tenv : Typing_env.t,
    Arms.use true discriminant argument_of_p argument_of_q = Some tenv
    ∧ made_from Arms.source_env tenv.
Proof.
  intros Hsame Hsome. rewrite Hsame.
  destruct (checked_use discriminant argument_of_p argument_of_q)
    as [ tenv | ] eqn:Hchecked; [ | discriminate Hsome ].
  exists tenv. split; [ reflexivity | ].
  exact (checked_use_made _ _ _ tenv Hchecked).
Qed.

Theorem arms_made_from_source : arms_made_from_source_statement.
Proof.
  split; [ exact arms_source_fits | ].
  intros use Hin. unfold arms_uses in Hin.
  repeat (destruct Hin as [ <- | Hin ];
          [ apply arms_use_made; vm_compute; reflexivity | ]).
  destruct Hin.
Qed.
