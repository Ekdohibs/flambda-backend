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


(* Proofs of the statements of FlambdaSimplifySwitchSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaControlFlowSpec FlambdaControlFlowProofs FlambdaClosuresSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs
  FlambdaMeetEnvProofs FlambdaTypeCreatorsSpec FlambdaTypeCreatorsProofs
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeetRenameSpec FlambdaMeetRenameProofs
  FlambdaSimplifySwitchSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesProofs
  FlambdaMeetClosuresProofs.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* What is made of the arms that are left                                   *)
(* ======================================================================== *)

Theorem switch_no_arms : switch_no_arms_statement.
Proof.
  intros scrutinee vars conts statics hp rd. apply step_eq_equivalent.
  rewrite step_switch, step_invalid.
  destruct (eval_simple vars scrutinee) as [v | ]; [ | reflexivity ].
  destruct (naked_immediate_of_value v); reflexivity.
Qed.

Theorem switch_of_arms_refines : switch_of_arms_refines_statement.
Proof.
  intros scrutinee arms.
  destruct arms as [ | [i ac] [ | arm arms]].
  - apply expr_equivalent_refines, switch_no_arms.
  - apply switch_same_arms. intros j arm [[= _ <-] | []]. reflexivity.
  - intros vars conts statics hp rd. apply refines_refl.
Qed.

(* ======================================================================== *)
(* Refinement where the static environment is valid                         *)
(* ======================================================================== *)

Lemma refines_under_valid (tenv : Typing_env.t) (e e' : expr) :
  expr_refines_under tenv e e' -> expr_refines_under_valid tenv e e'.
Proof. intros Hrefines vars conts statics hp rd _ Hsat. apply Hrefines, Hsat.
  Qed.

Lemma equivalent_under_valid (tenv : Typing_env.t) (e e' : expr) :
  expr_equivalent_under tenv e e' -> expr_equivalent_under_valid tenv e e'.
Proof.
  intros Hequivalent vars conts statics hp rd _ Hsat. apply Hequivalent, Hsat.
Qed.

Lemma equivalent_under_valid_refines (tenv : Typing_env.t) (e e' : expr) :
  expr_equivalent_under_valid tenv e e' -> expr_refines_under_valid tenv e e'.
Proof.
  intros Hequivalent vars conts statics hp rd Hvalid Hsat.
  apply equivalent_refines, Hequivalent; assumption.
Qed.

Theorem under_valid_weaken : under_valid_weaken_statement.
Proof.
  intros tenv e e'.
  split; [ apply refines_under_valid | ].
  split; [ apply equivalent_under_valid | ].
  apply equivalent_under_valid_refines.
Qed.

Lemma expr_refines_under_valid_trans (tenv : Typing_env.t)
    (e e' e'' : expr) :
  expr_refines_under_valid tenv e e' ->
  expr_refines_under_valid tenv e' e'' ->
  expr_refines_under_valid tenv e e''.
Proof.
  intros Hfirst Hsecond vars conts statics hp rd Hvalid Hsat.
  exact (refines_trans _ _ _ (Hfirst vars conts statics hp rd_ Hvalid Hsat)
           (Hsecond vars conts statics hp rd_ Hvalid Hsat)).
Qed.

(* ======================================================================== *)
(* Arms that cannot be taken                                                *)
(* ======================================================================== *)

(* The arm of a number that is kept is the arm that it had; a number
   that is not kept has no arm. *)
Lemma find_arm_keep_arms (keep : Z -> bool) (i : Z)
    (arms : list (Z * apply_cont)) :
  find_arm i (keep_arms keep arms)
  = if keep i then find_arm i arms else None.
Proof.
  unfold keep_arms. induction arms as [ | [j ac] arms IH]; simpl.
  - destruct (keep i); reflexivity.
  - destruct (Z.eqb_spec i j) as [-> | Hne].
    + destruct (keep j) eqn:Hkeep; simpl.
      * rewrite Z.eqb_refl. reflexivity.
      * exact IH.
    + destruct (keep j); simpl; [ | exact IH ].
      destruct (Z.eqb_spec i j) as [Heq | _]; [ contradiction | exact IH ].
Qed.

Lemma find_arm_some_in (i : Z) (arms : list (Z * apply_cont))
    (ac : apply_cont) :
  find_arm i arms = Some ac -> i ∈ map fst arms.
Proof.
  induction arms as [ | [j ac'] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb_spec i j) as [-> | _].
  - intros _. left. reflexivity.
  - intros Hfind. right. exact (IH Hfind).
Qed.

Lemma naked_immediate_of_value_some (v : value) (i : Z) :
  naked_immediate_of_value v = Some i -> v = Value_const (Naked_immediate i).
Proof.
  destruct v; try discriminate.
  match goal with c : const |- _ => destruct c; try discriminate end.
  intros [= ->]. reflexivity.
Qed.

Theorem switch_without_arms : switch_without_arms_statement.
Proof.
  intros tenv scrutinee keep arms Hnot_taken vars conts statics hp rd Hvalid
    Hsat.
  apply step_eq_equivalent. rewrite !step_switch.
  destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | reflexivity ].
  destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | reflexivity ].
  rewrite find_arm_keep_arms.
  destruct (keep i) eqn:Hkeep; [ reflexivity | ].
  destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ exfalso | reflexivity ].
  apply (Hnot_taken i (find_arm_some_in i arms ac Hfind) Hkeep statics vars
           Hvalid Hsat).
  rewrite Hv, (naked_immediate_of_value_some v i Hi). reflexivity.
Qed.

Theorem switch_canonical : switch_canonical_statement.
Proof.
  intros tenv scrutinee canonical arms Hcanonical vars conts statics hp rd Hsat.
  apply step_eq_equivalent. rewrite !step_switch.
  rewrite (canonical_in_terms_sound tenv statics vars scrutinee canonical
             Hsat Hcanonical).
  reflexivity.
Qed.

(* ======================================================================== *)
(* Simplify's test                                                          *)
(* ======================================================================== *)

(* A type that is found for a name is found at its kind. *)
Lemma find_at_its_kind (tenv : Typing_env.t) (n : name) (ty : flambda_type) :
  Typing_env.find tenv n None = Some ty ->
  Typing_env.find tenv n (Some (kind ty)) = Some ty.
Proof.
  unfold Typing_env.find, Typing_env.find_with_binding_time_and_mode'.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty0 btm] | ].
  - cbn [Typing_env.check_optional_kind_matches].
    destruct (Typing_env.is_bottom tenv); cbn [option_map fst];
      intros [= <-].
    + rewrite kind_bottom, kind_eqb_refl. reflexivity.
    + rewrite kind_eqb_refl. reflexivity.
  - destruct n as [x | sym]; [ discriminate | ].
    destruct (among sym (Typing_env.defined_symbols tenv));
      [ | discriminate ].
    cbn. intros [= <-]. reflexivity.
Qed.

(* What [type_simple_in_term_exn] answers: the alias type of an element
   that the aliases answer for the [Simple], at a kind of the
   [Simple]. *)
Lemma type_simple_in_term_exn_answer (tenv : Typing_env.t)
    (min_name_mode : option Name_mode.t) (s canonical : simple)
    (ty : flambda_type) :
  Typing_env.type_simple_in_term_exn tenv min_name_mode s
  = Some (Some (ty, canonical)) ->
  ∃ (k : flambda_kind) (name_mode min_mode : Name_mode.t),
    simple_of_kind tenv s k
    ∧ ty = alias_type_of k canonical
    ∧ Aliases.get_canonical_element_exn (Typing_env.binding_time_resolver tenv)
        (Typing_env.names_to_types tenv) (Typing_env.aliases tenv) s
        name_mode min_mode (Typing_env.min_binding_time tenv)
      = Some canonical.
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
            end) as [[ty' btm] | ] eqn:Hfound; [ | discriminate ].
  destruct (Aliases.get_canonical_element_exn _ _ _ _ _ _ _) as [alias | ]
    eqn:Hcanonical; [ | discriminate ].
  intros [= <- <-].
  exists (kind ty'), (Binding_time.With_name_mode.name_mode btm).
  eexists. split; [ | split; [ reflexivity | exact Hcanonical ] ].
  assert (Hname :
    ∀ n : name,
      Typing_env.find_with_binding_time_and_mode tenv n None
      = Some (ty', btm) ->
      simple_of_kind tenv (simple_of_name n) (kind ty')).
  { intros n. unfold Typing_env.find_with_binding_time_and_mode.
    destruct (Typing_env.find_with_binding_time_and_mode' tenv n None)
      as [[ty0 btm0] | ] eqn:Hfind; [ | discriminate ].
    intros [= <- _]. apply simple_of_kind_of_name. exists ty0.
    apply find_at_its_kind. unfold Typing_env.find. rewrite Hfind.
    reflexivity. }
  destruct s as [x | sym | c].
  - exact (Hname (Name_var x) Hfound).
  - exact (Hname (Name_symbol sym) Hfound).
  - injection Hfound as <- _. reflexivity.
Qed.

Theorem type_simple_in_term_fits : type_simple_in_term_fits_statement.
Proof.
  intros tenv min_name_mode s canonical ty Hfits Htype.
  destruct (type_simple_in_term_exn_answer tenv _ s canonical ty Htype)
    as [k [name_mode [min_mode [Hkind [-> Hcanonical]]]]].
  pose proof Hfits as [Hinvariant [_ [[_ [Haliases_kinded _]] _]]].
  pose proof Hinvariant as [Hwf _].
  pose proof (proj1 (typing_env_well_formed_iff tenv) Hwf) as Hconditions.
  destruct Hconditions as [_ [_ [Hconsistent _]]].
  pose proof (canonical_at_mode_aliased _ _ _ _ _ _ _ _ _ Hconsistent
                Hcanonical) as Haliased.
  assert (Hknown : Typing_env.mem_simple None tenv canonical = true).
  { destruct (simple_equal canonical s) eqn:Hequal.
    - apply simple_equal_decides_equality in Hequal. subst canonical.
      exact (simple_of_kind_mem_simple tenv s k Hinvariant Hkind).
    - assert (Hne : canonical <> s).
      { intros Heq. apply simple_equal_decides_equality in Heq.
        congruence. }
      destruct canonical as [x | sym | c]; [ | | reflexivity ].
      + exact (well_formed_mentioned_known tenv _ Hwf
                 (aliased_mentioned _ _ _ (Name_var x) Haliased Hne eq_refl)).
      + exact (well_formed_mentioned_known tenv _ Hwf
                 (aliased_mentioned _ _ _ (Name_symbol sym) Haliased Hne
                    eq_refl)). }
  destruct (mem_simple_of_some_kind tenv canonical Hknown) as [k' Hk'].
  rewrite <- (Haliases_kinded canonical s k' k Haliased Hk' Hkind).
  apply type_fits_alias; assumption.
Qed.

Theorem pruned_arm_not_taken : pruned_arm_not_taken_statement.
Proof.
  intros meet tenv s canonical ty i Hmeet Hfits Hmem Htype Hkept statics vars
    Hvalid [_ [valuation [Hsatisfied Hagrees]]] Heval.
  unfold arm_kept in Hkept.
  destruct (meet tenv ty (this_naked_immediate i)) as [[answer | ] | ]
    eqn:Hanswer; try discriminate.
  assert (Hvalue :
    eval_simple valuation s = Some (Value_const (Naked_immediate i))).
  { rewrite (agrees_eval_simple tenv valuation vars s Hagrees);
      [ exact Heval | ].
    intros x ->. exact Hmem. }
  pose proof (proj2 (type_simple_in_term_satisfied (fun _ => False) statics
                       valuation tenv (Some Name_mode.normal) s canonical ty
                       Hsatisfied Htype) _ Hvalue) as Hty.
  apply (Hmeet statics valuation tenv ty (this_naked_immediate i)
           (Value_const (Naked_immediate i)));
    [ | exact Hvalid | exact Hsatisfied | exact Hanswer | ].
  - split; [ exact Hfits | ].
    split; [ exact (type_simple_in_term_fits tenv _ s canonical ty Hfits
                      Htype)
           | ].
    split.
    + apply (type_fits_alias tenv (Naked_number Kind_naked_immediate)
               (Const (Naked_immediate i))); reflexivity.
    + rewrite <- (type_satisfied_kind _ _ _ _ Hty). reflexivity.
  - split; [ exact Hty | ]. apply this_const_satisfied. reflexivity.
Qed.

(* The arms that Simplify keeps are those whose test does not answer
   that the arm is dropped. *)
Lemma prune_arms_keep_arms (meet : meet_function) (tenv : Typing_env.t)
    (ty : flambda_type) :
  ∀ arms arms' : list (Z * apply_cont),
    prune_arms meet tenv ty arms = Some arms' ->
    arms'
    = keep_arms
        (fun i =>
           match arm_kept meet tenv ty i with
           | Some false => false
           | _ => true
           end)
        arms.
Proof.
  induction arms as [ | [i ac] arms IH]; intros arms'; simpl.
  - intros [= <-]. reflexivity.
  - destruct (arm_kept meet tenv ty i) as [kept | ] eqn:Hkept;
      [ | discriminate ].
    destruct (prune_arms meet tenv ty arms) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. rewrite (IH rest eq_refl). unfold keep_arms.
    cbn [filter fst]. destruct kept; reflexivity.
Qed.

Theorem prune_arms_equivalent : prune_arms_equivalent_statement.
Proof.
  intros meet tenv s canonical ty arms arms' Hmeet Hfits Hmem Htype Hprune.
  rewrite (prune_arms_keep_arms meet tenv ty arms arms' Hprune).
  apply switch_without_arms. intros i _ Hkeep.
  apply (pruned_arm_not_taken meet tenv s canonical ty i Hmeet Hfits Hmem
           Htype).
  destruct (arm_kept meet tenv ty i) as [[ | ] | ]; try discriminate.
  reflexivity.
Qed.

Theorem simplify_switch_refines : simplify_switch_refines_statement.
Proof.
  intros meet tenv s arms e Hmeet Hfits Hmem. unfold simplify_switch.
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty canonical] | ] | ] eqn:Htype; try discriminate.
  cbn.
  destruct (prune_arms meet tenv ty arms) as [arms' | ] eqn:Hprune;
    [ | discriminate ].
  intros [= <-].
  apply (expr_refines_under_valid_trans tenv _ (Switch s arms')).
  { apply equivalent_under_valid_refines.
    exact (prune_arms_equivalent meet tenv s canonical ty arms arms' Hmeet
             Hfits Hmem Htype Hprune). }
  apply (expr_refines_under_valid_trans tenv _ (Switch canonical arms')).
  { apply equivalent_under_valid_refines, equivalent_under_valid.
    apply switch_canonical.
    split; [ exact Hmem | exists ty; exact Htype ]. }
  apply refines_under_valid, expr_refines_under_weaken, switch_of_arms_refines.
Qed.

(* ======================================================================== *)
(* With the meet that the contract is proved of                             *)
(* ======================================================================== *)

Theorem meet_to_prove_bottom_sound : meet_to_prove_bottom_sound_statement.
Proof.
  intros join fuel Hjoin. apply meet_answers_bottom.
  exact (proj1 (FlambdaMeetClosuresProofs.meet_to_prove_sound join fuel Hjoin)).
Qed.

Theorem simplify_switch_of_meet_to_prove_refines :
  simplify_switch_of_meet_to_prove_refines_statement.
Proof.
  intros join fuel tenv scrutinee arms e Hjoin.
  apply simplify_switch_refines, meet_to_prove_bottom_sound, Hjoin.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Theorem arm_kept_rename : arm_kept_rename_statement.
Proof.
  intros meet r tenv ty i Hmeet. unfold arm_kept.
  rewrite <- (rename_type_this_const r (Naked_immediate i)) at 1.
  unfold this_naked_immediate. rewrite Hmeet.
  destruct (meet tenv ty (this_const (Naked_immediate i)))
    as [[[t tenv'] | ] | ]; reflexivity.
Qed.

Theorem prune_arms_rename : prune_arms_rename_statement.
Proof.
  intros meet r tenv ty arms Hmeet.
  induction arms as [ | [i ac] arms IH]; [ reflexivity | ].
  cbn [rename_arms map fst snd prune_arms].
  rewrite (arm_kept_rename meet r tenv ty i Hmeet).
  destruct (arm_kept meet tenv ty i) as [kept | ]; [ | reflexivity ].
  change (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    with (rename_arms r arms).
  rewrite IH.
  destruct (prune_arms meet tenv ty arms) as [rest | ]; [ | reflexivity ].
  destruct kept; reflexivity.
Qed.

Lemma switch_of_arms_rename (r : renaming) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  switch_of_arms (rename_simple r scrutinee) (rename_arms r arms)
  = rename_expr r (switch_of_arms scrutinee arms).
Proof. destruct arms as [ | [i ac] [ | arm arms]]; reflexivity. Qed.

Theorem simplify_switch_rename : simplify_switch_rename_statement.
Proof.
  intros meet r tenv s arms Hmeet. unfold simplify_switch.
  rewrite type_simple_in_term_exn_equivariant.
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty canonical] | ] | ]; try reflexivity.
  cbn [option_map fst snd].
  rewrite (prune_arms_rename meet r tenv ty arms Hmeet).
  destruct (prune_arms meet tenv ty arms) as [arms' | ]; [ | reflexivity ].
  cbn [option_map]. rewrite switch_of_arms_rename. reflexivity.
Qed.

Theorem simplify_switch_of_meet_to_prove_rename :
  simplify_switch_of_meet_to_prove_rename_statement.
Proof.
  intros join fuel r tenv s arms Hjoin.
  apply simplify_switch_rename, meet_to_prove_equivariant, Hjoin.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Definition no_elements {A : Type} (l : list A) : bool :=
  match l with
  | [] => true
  | _ :: _ => false
  end.

(* A test of the types of an environment: each is well formed, has no
   alias and no equation inside, and is not obviously bottom, and that
   of a symbol is of the kind of values. *)
Definition plain_type_b (n : name)
    (entry : flambda_type * Binding_time.With_name_mode.t) : bool :=
  type_well_formed (fst entry)
  && no_elements (aliases_with_kinds (fst entry))
  && no_elements (equation_names_with_kinds (fst entry))
  && negb (is_obviously_bottom (fst entry))
  && match n with
     | Name_var _ => true
     | Name_symbol _ => kind_eqb (kind (fst entry)) Value
     end.

(* With the invariant and the kinds of what is known to be equal, the
   test is enough for meets to be made in the environment. *)
Lemma plain_types_fits (tenv : Typing_env.t) :
  typing_env_invariant tenv ->
  aliases_kinded tenv ->
  Keyed.for_all_bindings_b name_eqb plain_type_b
    (Typing_env.names_to_types tenv)
  = true ->
  typing_env_fits tenv.
Proof.
  intros Hinvariant Haliases Hplain.
  assert (Hall :
    ∀ (n : name) (ty : flambda_type) (btm : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty, btm) ->
      type_well_formed ty = true
      ∧ aliases_with_kinds ty = []
      ∧ equation_names_with_kinds ty = []
      ∧ is_obviously_bottom ty = false
      ∧ (∀ s : symbol, n = Name_symbol s -> kind ty = Value)).
  { intros n ty btm Hlookup.
    pose proof (proj1 (for_all_names_b_iff
                         (fun n entry => plain_type_b n entry = true)
                         plain_type_b _ (fun _ _ _ => iff_refl _))
                  Hplain n (ty, btm) Hlookup) as Htest.
    unfold plain_type_b in Htest. cbn [fst] in Htest.
    rewrite !andb_true_iff in Htest.
    destruct Htest as [[[[Hwf Hno_alias] Hno_equation] Hnot_bottom] Hsymbol].
    split; [ exact Hwf | ].
    split; [ destruct (aliases_with_kinds ty); [ reflexivity | discriminate ]
           | ].
    split;
      [ destruct (equation_names_with_kinds ty);
          [ reflexivity | discriminate ]
      | ].
    split; [ apply negb_true_iff; exact Hnot_bottom | ].
    intros s ->. apply kind_eqb_eq. exact Hsymbol. }
  split; [ exact Hinvariant | ].
  split.
  { intros n ty btm Hlookup. exact (proj1 (Hall n ty btm Hlookup)). }
  split.
  { split; [ | split; [ exact Haliases | ] ].
    - intros n ty btm Hlookup s k Hin.
      rewrite (proj1 (proj2 (Hall n ty btm Hlookup))) in Hin. destruct Hin.
    - intros s ty btm Hlookup.
      exact (proj2 (proj2 (proj2 (proj2 (Hall _ ty btm Hlookup)))) s
               eq_refl). }
  split.
  { intros n ty btm Hlookup m k Hin.
    rewrite (proj1 (proj2 (proj2 (Hall n ty btm Hlookup)))) in Hin.
    destruct Hin. }
  intros _ n ty btm Hlookup.
  exact (proj1 (proj2 (proj2 (proj2 (Hall n ty btm Hlookup))))).
Qed.

(* The two statements of FlambdaTypingEnvRefinementSpec.v about a [Let]
   that the examples use, where the static environment is valid: a
   [Let] does not change it. The proofs are those of
   FlambdaTypingEnvRefinementProofs.v with the premise carried. They
   are to go if the notions of that file come to ask for a valid
   static environment. *)
Lemma let_typed_refines_under_valid (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named) (body body' : expr)
    (ty : flambda_type) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
  (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
     (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    exec_named statics hp rd vars defining_expr = Some (v, hp', rd') ->
    type_satisfied statics vars v ty) ->
  expr_refines_under_valid
    (Typing_env.replace_equation
       (Typing_env.add_variable_definition tenv x (kind ty) Name_mode.normal)
       (Name_var x) ty)
    body body' ->
  expr_refines_under_valid tenv (Let x declared defining_expr body)
    (Let x declared defining_expr body').
Proof.
  intros Hwf Hfresh Hfree Htype Hbody vars conts statics hp rd Hvalid Hsat.
  apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv _ ].
  pose proof (Htype statics hp hp' rd rd' vars v Hsat Hv) as Hty.
  apply Hbody; [ exact Hvalid | ].
  apply replace_equation_env_satisfies_strictly.
  - apply (add_variable_definition_in_scope tenv x _ x Hwf). left.
    reflexivity.
  - intros y Hin. apply (add_variable_definition_in_scope tenv x _ y Hwf).
    right. split; [ | exact (Hfree y Hin) ].
    exact (not_bound_not_in_scope tenv x y Hfresh (Hfree y Hin)).
  - intros v' Hlookup.
    pose proof (eq_trans (eq_sym Hlookup) (lookup_head x v vars)) as Heq.
    injection Heq as ->.
    apply (type_satisfied_coincide statics vars ((x, v) :: vars) v ty);
      [ | exact Hty ].
    intros y Hin. symmetry. apply lookup_other.
    exact (not_bound_not_in_scope tenv x y Hfresh (Hfree y Hin)).
  - apply add_variable_definition_env_satisfies_strictly;
      [ exact Hwf | exact Hfresh | | exact Hsat ].
    exact (type_satisfied_kind statics vars v ty Hty).
Qed.

Lemma let_simple_refines_under_valid (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (s : simple) (body body' : expr)
    (kind : flambda_kind) (result : Typing_env.add_alias_result) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  (∀ y : variable, s = Var y -> in_scope_in_terms tenv y) ->
  simple_of_kind tenv s kind ->
  Typing_env.add_alias
    (Typing_env.add_variable_definition tenv x kind Name_mode.normal) (Var x)
    s
  = Some (Ok result) ->
  expr_refines_under_valid (Typing_env.add_alias_result_t result) body body' ->
  expr_refines_under_valid tenv (Let x declared (Simple s) body)
    (Let x declared (Simple s) body').
Proof.
  intros Hwf Hfresh Hscope Hkind Hadd Hbody vars conts statics hp rd Hvalid
    Hsat.
  apply let_refines_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | reflexivity | intros
     v Hv _ ]. simpl in Hv.
  assert (Hother :
    eval_simple ((x, v) :: vars) s = eval_simple vars s).
  { destruct s as [y | sym | c]; [ | reflexivity.. ]. simpl.
    apply lookup_other.
    exact (not_bound_not_in_scope tenv x y Hfresh (Hscope y eq_refl)). }
  apply Hbody; [ exact Hvalid | ].
  apply (add_alias_env_satisfies_strictly
           (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
           statics ((x, v) :: vars) (Var x) s result).
  - intros y [[= <-] | ->];
      apply (add_variable_definition_in_scope tenv x kind _ Hwf);
      [ left; reflexivity | right ].
    split; [ | exact (Hscope y eq_refl) ].
    exact (not_bound_not_in_scope tenv x y Hfresh (Hscope y eq_refl)).
  - exact Hadd.
  - rewrite Hother, Hv. simpl. apply lookup_head.
  - apply add_variable_definition_env_satisfies_strictly;
      [ exact Hwf | exact Hfresh | | exact Hsat ].
    apply (simple_of_kind_value tenv statics vars s kind v);
      [ | exact Hscope | exact Hkind | exact Hv ].
    apply env_satisfies_strictly_env_satisfies. exact Hsat.
Qed.

(* The meet that the examples run has what is asked of a meet. *)
Lemma example_meet_bottom_sound : meet_bottom_sound example_meet.
Proof.
  apply meet_to_prove_bottom_sound.
  exact FlambdaMeetExamplesProofs.forgetful_join_sound.
Qed.

(* The environment of the first example: [i] defined and made equal to
   the constant. *)
Lemma example_no_arm_env_fits (result : Typing_env.add_alias_result) :
  Typing_env.add_alias
    (Typing_env.add_variable_definition Typing_env.create example_i
       (Naked_number Kind_naked_immediate) Name_mode.normal)
    (Var example_i) (Const (Naked_immediate 5))
  = Some (Ok result) ->
  typing_env_fits (Typing_env.add_alias_result_t result).
Proof.
  intros Hadd.
  assert (Hwf :
    typing_env_well_formed
      (Typing_env.add_variable_definition Typing_env.create example_i
         (Naked_number Kind_naked_immediate) Name_mode.normal)
    = true) by (vm_compute; reflexivity).
  pose proof (proj1 (typing_env_well_formed_iff _) Hwf) as Hconditions.
  destruct Hconditions as [_ [_ [Hconsistent _]]].
  assert (Haliases : aliases_kinded (Typing_env.add_alias_result_t result)).
  { apply (add_alias_aliases_kinded _ (Var example_i)
             (Const (Naked_immediate 5)) result
             (Naked_number Kind_naked_immediate) Hconsistent).
    - vm_compute. reflexivity.
    - vm_compute. reflexivity.
    - eexists. vm_compute. reflexivity.
    - reflexivity.
    - exact Hadd.
    - apply add_variable_definition_aliases_kinded.
      + vm_compute. reflexivity.
      + reflexivity.
      + apply create_aliases_kinded. }
  vm_compute in Hadd. injection Hadd as <-.
  apply plain_types_fits; [ | exact Haliases | vm_compute; reflexivity ].
  split; [ vm_compute; reflexivity | ].
  intros n ty Hlookup. vm_compute in Hlookup. discriminate.
Qed.

Theorem example_switch_no_arm : example_switch_no_arm_statement.
Proof.
  intros k0 k1.
  eapply (let_simple_refines_under_valid Typing_env.create example_i
            (Naked_number Kind_naked_immediate) (Const (Naked_immediate 5))
            _ _ (Naked_number Kind_naked_immediate)).
  - reflexivity.
  - reflexivity.
  - intros y [=].
  - reflexivity.
  - vm_compute. reflexivity.
  - apply (simplify_switch_refines example_meet);
      [ exact example_meet_bottom_sound | | | ].
    + apply example_no_arm_env_fits. vm_compute. reflexivity.
    + reflexivity.
    + vm_compute. reflexivity.
Qed.

(* The environment of the second example: [x] defined at the kind of
   values, and [i] defined with the type of the two booleans. *)
Definition example_comparison_env : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition
       (Typing_env.add_variable_definition Typing_env.create example_x Value
          Name_mode.normal)
       example_i (Naked_number Kind_naked_immediate) Name_mode.normal)
    (Name_var example_i) any_naked_bool.

Lemma example_comparison_env_fits : typing_env_fits example_comparison_env.
Proof.
  apply plain_types_fits; [ | | vm_compute; reflexivity ].
  - split; [ vm_compute; reflexivity | ].
    intros n ty Hlookup.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    vm_compute in Hlookup. destruct Hlookup as [[= <- <-] | []].
    eexists. vm_compute. reflexivity.
  - apply replace_equation_aliases_kinded.
    + eexists. vm_compute. reflexivity.
    + apply add_variable_definition_aliases_kinded.
      * vm_compute. reflexivity.
      * reflexivity.
      * apply add_variable_definition_aliases_kinded.
        -- vm_compute. reflexivity.
        -- reflexivity.
        -- apply create_aliases_kinded.
Qed.

(* A comparison that yields a boolean has no value but the two
   booleans. *)
Lemma int_comp_bool (statics : static_env) (kind : standard_int)
    (c : int_comparison) (vs : list value) (v : value) :
  eval_prim statics (Binary (Int_comp kind (Yielding_bool c))) vs = Some v ->
  v = Value_const (Naked_immediate 0) ∨ v = Value_const (Naked_immediate 1).
Proof.
  destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
  cbn [eval_prim eval_binary_primitive].
  destruct (int_of_value kind v1) as [a | ]; [ | discriminate ].
  destruct (int_of_value kind v2) as [b | ]; [ | discriminate ].
  cbn. intros [= <-].
  destruct (eval_int_comparison (int_width kind) c a b); auto.
Qed.

Theorem example_switch_comparison : example_switch_comparison_statement.
Proof.
  intros k0 k1 k2.
  apply (let_typed_refines_under_valid _ example_i _ _ _ _ any_naked_bool).
  - vm_compute. reflexivity.
  - reflexivity.
  - intros y [].
  - intros statics hp hp' rd rd' vars v _ Hv. apply any_bool_satisfied.
    apply exec_named_pure in Hv as (Hv & _ & _); [ | reflexivity ].
    cbn [eval_named] in Hv.
    destruct (eval_simples vars _) as [vs | ]; [ | discriminate ].
    exact (int_comp_bool statics _ _ vs v Hv).
  - apply (simplify_switch_refines example_meet);
      [ exact example_meet_bottom_sound | | | ].
    + exact example_comparison_env_fits.
    + reflexivity.
    + vm_compute. reflexivity.
Qed.
