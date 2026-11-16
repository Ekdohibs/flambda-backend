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


(* Proofs of the statements of FlambdaTypingEnvSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* Kinds                                                                    *)
(* ======================================================================== *)

Lemma kind_bottom (k : flambda_kind) : kind (bottom k) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma kind_unknown (k : flambda_kind) : kind (unknown k) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma kind_alias_type_of (k : flambda_kind) (s : simple) :
  kind (alias_type_of k s) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

(* The type that the compiler gives a constant is of its kind. *)
Lemma kind_type_for_const (c : const) :
  kind (type_for_const c) = kind_of_const c.
Proof.
  symmetry.
  exact (type_satisfied_kind empty_static_env [] (Value_const c)
           (type_for_const c)
           (type_for_const_satisfied empty_static_env [] c)).
Qed.

(* ======================================================================== *)
(* Exempt names                                                             *)
(* ======================================================================== *)

Theorem has_type_unless_kind : has_type_unless_kind_statement.
Proof.
  intros exempt statics valuation v ty [[_ Hkind] | Hty]; [ exact Hkind | ].
  exact (type_satisfied_kind statics valuation v ty Hty).
Qed.

Theorem has_type_unless_not_exempt : has_type_unless_not_exempt_statement.
Proof.
  intros exempt statics valuation v ty [[Hexempt _] | Hty] Hnot_exempt;
    [ contradiction | exact Hty ].
Qed.

(* A value that has a type is what is asked of the value of a name of
   that type. *)
Lemma type_has_type_unless (exempt : Prop) (statics : static_env)
    (valuation : env) (v : value) (ty : flambda_type) :
  type_satisfied statics valuation v ty ->
  has_type_unless exempt statics valuation v ty.
Proof. intros Hty. right. exact Hty. Qed.

Lemma has_type_unless_weaken (exempt exempt' : Prop) (statics : static_env)
    (valuation : env) (v : value) (ty : flambda_type) :
  (exempt -> exempt') ->
  has_type_unless exempt statics valuation v ty ->
  has_type_unless exempt' statics valuation v ty.
Proof.
  intros Hweaken [[Hexempt Hkind] | Hty]; [ left | right; exact Hty ].
  split; [ exact (Hweaken Hexempt) | exact Hkind ].
Qed.

(* The value of a symbol is the symbol, in every valuation. *)
Lemma value_of_name_symbol (valuation : env) (s : symbol) :
  value_of_name valuation (Name_symbol s) = Some (Value_symbol s).
Proof. reflexivity. Qed.

Lemma value_of_name_var (valuation : env) (x : variable) :
  value_of_name valuation (Name_var x) = lookup x valuation.
Proof. reflexivity. Qed.

(* Whether a name holds poison depends on its value alone. *)
Lemma holds_poison_coincide (valuation valuation' : env) (n : name) :
  value_of_name valuation n = value_of_name valuation' n ->
  holds_poison valuation n ->
  holds_poison valuation' n.
Proof.
  intros Hsame [w [Hw Hpoison]]. exists w.
  split; [ rewrite <- Hsame; exact Hw | exact Hpoison ].
Qed.

(* The scrutinee of a relation of a type, if it is a variable, is a
   variable of the type. *)
Lemma relation_scrutinee_free (head : head_of_kind_naked_immediate)
    (relation : Relation.t) (y : variable) :
  (relation, Name_var y) ∈ inverse_relations head ->
  y ∈ free_variables (Type_naked_immediate (Ok (No_alias head))).
Proof.
  intros Hin. simpl. unfold free_variables_of_naked_immediate.
  apply in_flat_map. exists (relation, Name_var y).
  split; [ exact Hin | left; reflexivity ].
Qed.

(* Valuations that give the variables of a type the same values agree on
   whether the type is a relation on a name that holds poison. *)
Lemma relation_on_poison_coincide (valuation valuation' : env)
    (ty : flambda_type) :
  (∀ y : variable,
    y ∈ free_variables ty -> lookup y valuation = lookup y valuation') ->
  relation_on_poison valuation ty ->
  relation_on_poison valuation' ty.
Proof.
  intros Hsame [head [relation [scrutinee [-> [Hin [Hwhich Hpoison]]]]]].
  exists head, relation, scrutinee. split; [ reflexivity | ].
  split; [ exact Hin | ]. split; [ exact Hwhich | ].
  apply (holds_poison_coincide valuation valuation' scrutinee);
    [ | exact Hpoison ].
  destruct scrutinee as [y | sym]; [ | reflexivity ].
  rewrite !value_of_name_var. apply Hsame.
  exact (relation_scrutinee_free head relation y Hin).
Qed.

(* ======================================================================== *)
(* Reading the lookup functions back                                        *)
(* ======================================================================== *)

(* What [find_with_binding_time_and_mode'] answers, in an environment
   that is not bottom: the entry of the name, or the type of a symbol
   that is defined and has none. *)
Lemma find_with_binding_time_and_mode'_cases (tenv : Typing_env.t) (n : name)
    (kind : option flambda_kind) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) :
  Typing_env.is_bottom tenv = false ->
  Typing_env.find_with_binding_time_and_mode' tenv n kind
  = Some (ty, binding_time_and_mode) ->
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
  = Some (ty, binding_time_and_mode)
  ∨ ((∃ s : symbol, n = Name_symbol s)
     ∧ (ty, binding_time_and_mode) = Typing_env.initial_symbol_type).
Proof.
  intros Hbottom. unfold Typing_env.find_with_binding_time_and_mode'.
  rewrite Hbottom.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty' binding_time_and_mode'] | ] eqn:Hlookup.
  - destruct (Typing_env.check_optional_kind_matches ty' kind);
      [ | discriminate ].
    intros [= <- <-]. left. reflexivity.
  - destruct n as [x | s]; [ discriminate | ].
    destruct (among s (Typing_env.defined_symbols tenv)
              && Typing_env.check_optional_kind_matches
                   (fst Typing_env.initial_symbol_type) kind);
      [ | discriminate ].
    intros [= <- <-]. right. split; [ exists s; reflexivity | reflexivity ].
Qed.

(* A valuation that satisfies an environment gives the name its type,
   whichever of the two it is. *)
Lemma find_with_binding_time_and_mode'_satisfied (exempt : name -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t) (n : name)
    (kind : option flambda_kind) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) (v : value) :
  typing_env_satisfied_but exempt statics valuation tenv ->
  Typing_env.find_with_binding_time_and_mode' tenv n kind
  = Some (ty, binding_time_and_mode) ->
  value_of_name valuation n = Some v ->
  has_type_unless (exempt n) statics valuation v ty.
Proof.
  intros [Hbottom [Htypes _]] Hfind Hvalue.
  apply find_with_binding_time_and_mode'_cases in Hfind; [ | exact Hbottom ].
  destruct Hfind as [Hlookup | [[s ->] [= -> ->]]].
  - exact (Htypes n ty binding_time_and_mode v Hlookup Hvalue).
  - rewrite value_of_name_symbol in Hvalue. injection Hvalue as <-.
    apply type_has_type_unless, any_non_null_value_satisfied.
    split; [ reflexivity | discriminate ].
Qed.

Lemma find_with_binding_time_and_mode_satisfied (exempt : name -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t) (n : name)
    (kind : option flambda_kind) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) (v : value) :
  typing_env_satisfied_but exempt statics valuation tenv ->
  Typing_env.find_with_binding_time_and_mode tenv n kind
  = Some (ty, binding_time_and_mode) ->
  value_of_name valuation n = Some v ->
  has_type_unless (exempt n) statics valuation v ty.
Proof.
  intros Hsatisfied. unfold Typing_env.find_with_binding_time_and_mode.
  destruct (Typing_env.find_with_binding_time_and_mode' tenv n kind)
    as [[ty' binding_time_and_mode'] | ] eqn:Hfind; [ | discriminate ].
  intros [= <- _].
  exact (find_with_binding_time_and_mode'_satisfied exempt statics valuation
           tenv n kind ty' binding_time_and_mode' v Hsatisfied Hfind).
Qed.

(* [mem] with the normal name mode, of a variable: its entry has the
   normal name mode in scope. *)
Lemma in_scope_in_terms_lookup (tenv : Typing_env.t) (x : variable) :
  in_scope_in_terms tenv x
  <-> ∃ (ty : flambda_type)
        (binding_time_and_mode : Binding_time.With_name_mode.t),
        Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
        = Some (ty, binding_time_and_mode)
        ∧ Binding_time.With_name_mode.scoped_name_mode binding_time_and_mode
            (Typing_env.min_binding_time tenv)
          = Name_mode.Normal.
Proof.
  unfold in_scope_in_terms, Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv))
    as [[ty binding_time_and_mode] | ].
  - split.
    + intros Hmode. exists ty, binding_time_and_mode.
      split; [ reflexivity | ].
      destruct (Binding_time.With_name_mode.scoped_name_mode
                  binding_time_and_mode (Typing_env.min_binding_time tenv));
        [ reflexivity | discriminate | discriminate ].
    + intros [ty' [binding_time_and_mode' [[= <- <-] ->]]]. reflexivity.
  - split; [ discriminate | ].
    intros [ty [binding_time_and_mode [Hlookup _]]]. discriminate.
Qed.

(* ======================================================================== *)
(* What every environment satisfies, and what none does                     *)
(* ======================================================================== *)

(* What [Aliases.get_canonical_element_exn] answers for the normal name
   mode, if it is a variable: the element asked about, or a variable in
   scope with the normal name mode. *)
Lemma get_canonical_element_exn_normal {A : Type}
    (resolver : name -> Binding_time.With_name_mode.t)
    (map : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (a : Aliases.t) (element : simple) (elt_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) (y : variable) :
  Aliases.get_canonical_element_exn resolver map a element elt_name_mode
    Name_mode.Normal min_binding_time
  = Some (Var y) ->
  element = Var y
  ∨ ∃ (data : A) (binding_time_and_mode : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb (Name_var y) map
      = Some (data, binding_time_and_mode)
      ∧ Binding_time.With_name_mode.scoped_name_mode binding_time_and_mode
          min_binding_time
        = Name_mode.Normal.
Proof.
  intros Hanswer. apply canonical_at_mode_name_mode in Hanswer.
  destruct Hanswer as [[-> _] | [c [Hcompare Hc]]];
    [ left; reflexivity | right ].
  unfold Aliases.compute_name_mode, Aliases.binding_time_and_name_mode in *.
  destruct (Keyed.lookup name_eqb (Name_var y) map)
    as [[data binding_time_and_mode] | ].
  - exists data, binding_time_and_mode. split; [ reflexivity | ].
    destruct (Binding_time.With_name_mode.scoped_name_mode
                binding_time_and_mode min_binding_time);
      [ reflexivity | | ]; simpl in Hcompare; injection Hcompare as <-; lia.
  - exfalso. revert Hcompare.
    unfold Binding_time.With_name_mode.scoped_name_mode. simpl.
    destruct (min_binding_time <=? Binding_time.imported_variables)%Z;
      simpl; intros [= <-]; lia.
Qed.

(* ======================================================================== *)
(* How the notions compare                                                  *)
(* ======================================================================== *)

Theorem satisfied_but_weaken : satisfied_but_weaken_statement.
Proof.
  intros exempt exempt' statics valuation tenv Hweaken
    [Hbottom [Htypes Hrespected]].
  split; [ exact Hbottom | ]. split; [ | exact Hrespected ].
  intros n ty binding_time_and_mode v Hlookup Hvalue.
  exact (has_type_unless_weaken _ _ statics valuation v ty (Hweaken n)
           (Htypes n ty binding_time_and_mode v Hlookup Hvalue)).
Qed.

Theorem satisfied_strictly_satisfied : satisfied_strictly_satisfied_statement.
Proof.
  intros statics valuation tenv.
  apply satisfied_but_weaken. intros n [].
Qed.

Theorem satisfied_no_poison_strictly
    : satisfied_no_poison_strictly_statement.
Proof.
  intros statics valuation tenv Hno_poison.
  apply satisfied_but_weaken.
  intros n [Hpoison | [ty [btm [_ [head [relation [scrutinee Hrelation]]]]]]].
  - exact (Hno_poison n Hpoison).
  - exact (Hno_poison scrutinee (proj2 (proj2 (proj2 Hrelation)))).
Qed.

Theorem env_satisfies_strictly_env_satisfies
    : env_satisfies_strictly_env_satisfies_statement.
Proof.
  intros tenv statics vars [Hbound [valuation [Hsatisfied Hagrees]]].
  split; [ exact Hbound | ]. exists valuation.
  split; [ | exact Hagrees ].
  exact (satisfied_strictly_satisfied statics valuation tenv Hsatisfied).
Qed.

(* ======================================================================== *)
(* What satisfaction depends on                                             *)
(* ======================================================================== *)

(* Valuations that give the variables that an environment mentions the
   same values give the names that it has a type for the same values. *)
Lemma mentions_value_of_name (tenv : Typing_env.t)
    (valuation valuation' : env) (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) :
  (∀ x : variable,
    typing_env_mentions tenv x ->
    lookup x valuation = lookup x valuation') ->
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
  = Some (ty, binding_time_and_mode) ->
  value_of_name valuation n = value_of_name valuation' n.
Proof.
  intros Hsame Hlookup. destruct n as [x | s]; [ | reflexivity ].
  rewrite !value_of_name_var. apply Hsame. left.
  exists ty, binding_time_and_mode. exact Hlookup.
Qed.

Theorem satisfied_but_coincide : satisfied_but_coincide_statement.
Proof.
  intros exempt statics valuation valuation' tenv Hsame
    [Hbottom [Htypes Hrespected]].
  split; [ exact Hbottom | ]. split.
  - intros n ty binding_time_and_mode v Hlookup Hvalue.
    rewrite <- (mentions_value_of_name tenv valuation valuation' n ty
                  binding_time_and_mode Hsame Hlookup) in Hvalue.
    destruct (Htypes n ty binding_time_and_mode v Hlookup Hvalue)
      as [Hexempt | Hty]; [ left; exact Hexempt | right ].
    apply (proj1 (type_satisfied_coincide statics valuation valuation' v ty
                    (fun x Hx =>
                       Hsame x
                         (or_intror
                            (or_introl
                               (ex_intro _ n
                                  (ex_intro _ ty
                                     (ex_intro _ binding_time_and_mode
                                        (conj Hlookup Hx))))))))).
    exact Hty.
  - apply (respected_coincide (Typing_env.aliases tenv) valuation valuation');
      [ | exact Hrespected ].
    intros n Hmentioned. destruct n as [x | s]; [ | reflexivity ].
    rewrite !value_of_name_var. apply Hsame. right. right. exact Hmentioned.
Qed.

Theorem typing_env_satisfied_coincide
    : typing_env_satisfied_coincide_statement.
Proof.
  intros statics valuation valuation' tenv Hsame Hsatisfied.
  apply (satisfied_but_coincide (exempt_by_poison valuation tenv) statics
           valuation valuation' tenv Hsame) in Hsatisfied.
  destruct Hsatisfied as [Hbottom [Htypes Hrespected]].
  split; [ exact Hbottom | ]. split; [ | exact Hrespected ].
  intros n ty binding_time_and_mode v Hlookup Hvalue.
  apply (has_type_unless_weaken (exempt_by_poison valuation tenv n));
    [ | exact (Htypes n ty binding_time_and_mode v Hlookup Hvalue) ].
  intros [Hpoison | [ty' [btm [Hlookup' Hrelation]]]].
  - left. apply (holds_poison_coincide valuation valuation' n);
      [ | exact Hpoison ].
    exact (mentions_value_of_name tenv valuation valuation' n ty
             binding_time_and_mode Hsame Hlookup).
  - right. exists ty', btm. split; [ exact Hlookup' | ].
    apply (relation_on_poison_coincide valuation valuation' ty');
      [ | exact Hrelation ].
    intros y Hy. apply Hsame. right. left. exists n, ty', btm.
    split; [ exact Hlookup' | exact Hy ].
Qed.

(* The variables in scope, and what the machine and a valuation have
   for them, when the machine changes outside them. *)
Lemma bound_in_terms_coincide (tenv : Typing_env.t) (vars vars' : env) :
  (∀ x : variable,
    in_scope_in_terms tenv x -> lookup x vars = lookup x vars') ->
  bound_in_terms tenv vars ->
  bound_in_terms tenv vars'.
Proof.
  intros Hcoincide Hbound x Hscope.
  rewrite <- (Hcoincide x Hscope). exact (Hbound x Hscope).
Qed.

Lemma agrees_in_terms_coincide (tenv : Typing_env.t)
    (valuation vars vars' : env) :
  (∀ x : variable,
    in_scope_in_terms tenv x -> lookup x vars = lookup x vars') ->
  agrees_in_terms tenv valuation vars ->
  agrees_in_terms tenv valuation vars'.
Proof.
  intros Hcoincide Hagrees x Hscope.
  rewrite (Hagrees x Hscope). exact (Hcoincide x Hscope).
Qed.

Theorem env_satisfies_coincide : env_satisfies_coincide_statement.
Proof.
  intros tenv statics vars vars' Hcoincide
    [Hbound [valuation [Hsatisfied Hagrees]]].
  split.
  - exact (bound_in_terms_coincide tenv vars vars' Hcoincide Hbound).
  - exists valuation. split; [ exact Hsatisfied | ].
    exact (agrees_in_terms_coincide tenv valuation vars vars' Hcoincide
             Hagrees).
Qed.

Theorem env_satisfies_strictly_coincide
    : env_satisfies_strictly_coincide_statement.
Proof.
  intros tenv statics vars vars' Hcoincide
    [Hbound [valuation [Hsatisfied Hagrees]]].
  split.
  - exact (bound_in_terms_coincide tenv vars vars' Hcoincide Hbound).
  - exists valuation. split; [ exact Hsatisfied | ].
    exact (agrees_in_terms_coincide tenv valuation vars vars' Hcoincide
             Hagrees).
Qed.

Theorem satisfied_strictly_env_satisfies
    : satisfied_strictly_env_satisfies_statement.
Proof.
  intros tenv statics vars Hbound Hsatisfied.
  split; [ exact Hbound | ]. exists vars.
  split; [ exact Hsatisfied | intros x _; reflexivity ].
Qed.

Theorem typing_env_satisfied_env_satisfies
    : typing_env_satisfied_env_satisfies_statement.
Proof.
  intros tenv statics vars Hbound Hsatisfied.
  split; [ exact Hbound | ]. exists vars.
  split; [ exact Hsatisfied | intros x _; reflexivity ].
Qed.

(* ======================================================================== *)
(* What every environment satisfies, and what none does                     *)
(* ======================================================================== *)

Theorem create_satisfied : create_satisfied_statement.
Proof.
  intros statics vars. apply satisfied_strictly_env_satisfies.
  - intros x Hscope. discriminate.
  - split; [ reflexivity | ]. split.
    + intros n ty binding_time_and_mode v Hlookup. discriminate.
    + apply empty_respected.
Qed.

Theorem bottom_not_satisfied_but : bottom_not_satisfied_but_statement.
Proof.
  intros exempt statics valuation tenv Hbottom [Hnot_bottom _]. congruence.
Qed.

Theorem bottom_not_satisfied : bottom_not_satisfied_statement.
Proof.
  intros tenv statics vars Hbottom [_ [valuation [Hsatisfied _]]].
  exact (bottom_not_satisfied_but _ statics valuation tenv Hbottom
           Hsatisfied).
Qed.

Theorem make_bottom_not_satisfied : make_bottom_not_satisfied_statement.
Proof.
  intros tenv statics vars. apply bottom_not_satisfied. reflexivity.
Qed.

(* ======================================================================== *)
(* Looking a name up                                                        *)
(* ======================================================================== *)

Theorem find_satisfied : find_satisfied_statement.
Proof.
  intros exempt statics valuation tenv n kind ty v Hsatisfied.
  unfold Typing_env.find.
  destruct (Typing_env.find_with_binding_time_and_mode' tenv n kind)
    as [[ty' binding_time_and_mode] | ] eqn:Hfind; [ | discriminate ].
  intros [= <-].
  exact (find_with_binding_time_and_mode'_satisfied exempt statics valuation
           tenv n kind ty' binding_time_and_mode v Hsatisfied Hfind).
Qed.

Theorem find_satisfied_strictly : find_satisfied_strictly_statement.
Proof.
  intros statics valuation tenv n kind ty v Hsatisfied Hfind Hvalue.
  apply (has_type_unless_not_exempt False statics valuation v ty);
    [ | intros [] ].
  exact (find_satisfied (fun _ => False) statics valuation tenv n kind ty v
           Hsatisfied Hfind Hvalue).
Qed.

Theorem alias_has_type_found : alias_has_type_found_statement.
Proof.
  intros statics valuation tenv n kind ty k v Hsatisfied Hfind Halias.
  apply alias_satisfied in Halias. destruct Halias as [Hvalue _].
  exact (find_satisfied_strictly statics valuation tenv n kind ty v
           Hsatisfied Hfind Hvalue).
Qed.

(* In an environment that is not bottom, the type found for a name
   that has an entry is the type of the entry. *)
Lemma find_lookup (tenv : Typing_env.t) (n : name)
    (kind : option flambda_kind) (ty ty' : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) :
  Typing_env.is_bottom tenv = false ->
  Typing_env.find tenv n kind = Some ty ->
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
  = Some (ty', binding_time_and_mode) ->
  ty = ty'.
Proof.
  intros Hbottom Hfind Hlookup. revert Hfind.
  unfold Typing_env.find, Typing_env.find_with_binding_time_and_mode'.
  rewrite Hlookup, Hbottom.
  destruct (Typing_env.check_optional_kind_matches ty' kind);
    [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

Theorem find_not_poison : find_not_poison_statement.
Proof.
  intros statics valuation tenv n kind ty v Hsatisfied Hfind Hvalue
    Hnot_poison Hnot_relation.
  apply (has_type_unless_not_exempt (exempt_by_poison valuation tenv n)
           statics valuation v ty).
  - exact (find_satisfied (exempt_by_poison valuation tenv) statics valuation
             tenv n kind ty v Hsatisfied Hfind Hvalue).
  - intros [[w [Hw Hpoison]] | [ty' [btm [Hlookup Hrelation]]]].
    + rewrite Hvalue in Hw. injection Hw as <-.
      exact (Hnot_poison Hpoison).
    + rewrite (find_lookup tenv n kind ty ty' btm (proj1 Hsatisfied) Hfind
                 Hlookup) in Hnot_relation.
      exact (Hnot_relation Hrelation).
Qed.

Theorem find_kind : find_kind_statement.
Proof.
  intros tenv n k ty. unfold Typing_env.find,
    Typing_env.find_with_binding_time_and_mode'.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty' binding_time_and_mode] | ].
  - simpl. destruct (kind_eqb k (kind ty')) eqn:Hkind; [ | discriminate ].
    apply kind_eqb_eq in Hkind.
    destruct (Typing_env.is_bottom tenv); intros [= <-].
    + rewrite kind_bottom. symmetry. exact Hkind.
    + symmetry. exact Hkind.
  - destruct n as [x | s]; [ discriminate | ].
    destruct (among s (Typing_env.defined_symbols tenv)); [ | discriminate ].
    simpl. destruct (kind_eqb k Value) eqn:Hkind; [ | discriminate ].
    apply kind_eqb_eq in Hkind. intros [= <-]. symmetry. exact Hkind.
Qed.

Theorem find_cases : find_cases_statement.
Proof.
  intros tenv n kind ty.
  unfold Typing_env.find, Typing_env.find_with_binding_time_and_mode'.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty' binding_time_and_mode] | ].
  - destruct (Typing_env.check_optional_kind_matches ty' kind);
      [ | discriminate ].
    destruct (Typing_env.is_bottom tenv); intros [= <-].
    + right. left. eexists. reflexivity.
    + left. eexists. reflexivity.
  - destruct n as [x | s]; [ discriminate | ].
    destruct (among s (Typing_env.defined_symbols tenv)
              && Typing_env.check_optional_kind_matches
                   (fst Typing_env.initial_symbol_type) kind);
      [ | discriminate ].
    intros [= <-]. right. right. reflexivity.
Qed.

Theorem mem_find : mem_find_statement.
Proof.
  intros tenv min_name_mode n. unfold Typing_env.mem, Typing_env.find,
    Typing_env.find_with_binding_time_and_mode'.
  destruct n as [x | s].
  - destruct (Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types tenv))
      as [[ty binding_time_and_mode] | ]; [ | discriminate ].
    intros _. simpl.
    destruct (Typing_env.is_bottom tenv); eexists; reflexivity.
  - intros Hdefined.
    destruct (Keyed.lookup name_eqb (Name_symbol s)
                (Typing_env.names_to_types tenv))
      as [[ty binding_time_and_mode] | ].
    + simpl. destruct (Typing_env.is_bottom tenv); eexists; reflexivity.
    + rewrite Hdefined. simpl. eexists. reflexivity.
Qed.

(* A valuation that agrees with the machine gives a variable in scope
   the value that the machine has. *)
Lemma agrees_value_of_name (tenv : Typing_env.t) (valuation vars : env)
    (x : variable) :
  agrees_in_terms tenv valuation vars ->
  in_scope_in_terms tenv x ->
  value_of_name valuation (Name_var x) = lookup x vars.
Proof. intros Hagrees Hscope. exact (Hagrees x Hscope). Qed.

(* A type whose variables are in scope means the same under a valuation
   and under the variables of the machine that it agrees with. *)
Lemma type_satisfied_agrees (tenv : Typing_env.t) (statics : static_env)
    (valuation vars : env) (v : value) (ty : flambda_type) :
  agrees_in_terms tenv valuation vars ->
  (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
  type_satisfied statics valuation v ty ->
  type_satisfied statics vars v ty.
Proof.
  intros Hagrees Hfree.
  apply (proj1 (type_satisfied_coincide statics valuation vars v ty
                  (fun y Hy => Hagrees y (Hfree y Hy)))).
Qed.

Theorem in_scope_in_terms_bound : in_scope_in_terms_bound_statement.
Proof.
  intros tenv statics vars x ty [Hbound [valuation [Hsatisfied Hagrees]]]
    Hscope Hfind.
  destruct (Hbound x Hscope) as [v Hv]. exists v.
  split; [ exact Hv | ].
  apply (has_type_unless_kind
           (exempt_by_poison valuation tenv (Name_var x)) statics valuation v
           ty).
  apply (find_satisfied (exempt_by_poison valuation tenv) statics valuation
           tenv (Name_var x) None ty v Hsatisfied Hfind).
  rewrite (agrees_value_of_name tenv valuation vars x Hagrees Hscope).
  exact Hv.
Qed.

Theorem find_variable_in_terms_strictly
    : find_variable_in_terms_strictly_statement.
Proof.
  intros tenv statics vars x kind ty v [_ [valuation [Hsatisfied Hagrees]]]
    Hscope Hfind Hfree Hv.
  apply (type_satisfied_agrees tenv statics valuation vars v ty Hagrees
           Hfree).
  apply (find_satisfied_strictly statics valuation tenv (Name_var x) kind ty
           v Hsatisfied Hfind).
  rewrite (agrees_value_of_name tenv valuation vars x Hagrees Hscope).
  exact Hv.
Qed.

Theorem find_variable_in_terms : find_variable_in_terms_statement.
Proof.
  intros tenv statics vars x kind ty v [_ [valuation [Hsatisfied Hagrees]]]
    Hscope Hfind Hfree Hv Hnot_poison Hnot_relation.
  apply (type_satisfied_agrees tenv statics valuation vars v ty Hagrees
           Hfree).
  apply (find_not_poison statics valuation tenv (Name_var x) kind ty v
           Hsatisfied Hfind); [ | exact Hnot_poison | ].
  - rewrite (agrees_value_of_name tenv valuation vars x Hagrees Hscope).
    exact Hv.
  - intros Hrelation. apply Hnot_relation.
    apply (relation_on_poison_coincide valuation vars ty);
      [ | exact Hrelation ].
    intros y Hy. exact (Hagrees y (Hfree y Hy)).
Qed.

Theorem find_symbol_in_terms : find_symbol_in_terms_statement.
Proof.
  intros tenv statics vars s kind ty [_ [valuation [Hsatisfied Hagrees]]]
    Hfind Hfree.
  apply (type_satisfied_agrees tenv statics valuation vars (Value_symbol s)
           ty Hagrees Hfree).
  destruct (find_satisfied (exempt_by_poison valuation tenv) statics
              valuation tenv (Name_symbol s) kind ty (Value_symbol s)
              Hsatisfied Hfind eq_refl) as [[Hexempt Hkind] | Hty];
    [ exfalso | exact Hty ].
  destruct Hexempt
    as [[w [Hw [k Hpoison]]] | [ty' [btm [Hlookup Hrelation]]]].
  - rewrite value_of_name_symbol in Hw. injection Hw as <-. discriminate.
  - rewrite (find_lookup tenv (Name_symbol s) kind ty ty' btm
               (proj1 Hsatisfied) Hfind Hlookup) in Hkind.
    destruct Hrelation as [head [relation [scrutinee [-> _]]]].
    discriminate.
Qed.

(* ======================================================================== *)
(* Canonical elements                                                       *)
(* ======================================================================== *)

Theorem canonical_ignoring_name_mode_satisfied
    : canonical_ignoring_name_mode_satisfied_statement.
Proof.
  intros exempt statics valuation tenv s [_ [_ Hrespected]].
  destruct s as [x | s | c]; [ | | reflexivity ].
  - exact (respected_canonical (Typing_env.aliases tenv) valuation
             (Name_var x) Hrespected).
  - exact (respected_canonical (Typing_env.aliases tenv) valuation
             (Name_symbol s) Hrespected).
Qed.

Theorem get_canonical_simple_satisfied
    : get_canonical_simple_satisfied_statement.
Proof.
  intros exempt statics valuation tenv min_name_mode
    name_mode_of_existing_simple s canonical [_ [_ Hrespected]].
  unfold Typing_env.get_canonical_simple_exn.
  destruct (match name_mode_of_existing_simple with
            | Some name_mode => Some name_mode
            | None =>
              option_map Binding_time.With_name_mode.name_mode
                (Typing_env.binding_time_and_mode_of_simple tenv s)
            end) as [name_mode_simple | ]; [ | discriminate ].
  intros [= Hcanonical].
  exact (respected_canonical_at_mode _ _ _ _ valuation _ _ _ _ _ Hrespected
           Hcanonical).
Qed.

Theorem get_alias_then_canonical_simple_satisfied
    : get_alias_then_canonical_simple_satisfied_statement.
Proof.
  intros exempt statics valuation tenv min_name_mode
    name_mode_of_existing_simple typ canonical v Hsatisfied.
  unfold Typing_env.get_alias_then_canonical_simple_exn.
  destruct (get_alias_opt typ) as [s | ] eqn:Halias; [ | discriminate ].
  intros Hcanonical Hty.
  rewrite (get_canonical_simple_satisfied exempt statics valuation tenv
             min_name_mode name_mode_of_existing_simple s canonical
             Hsatisfied Hcanonical).
  apply (get_alias_satisfied statics valuation v typ s Halias) in Hty.
  exact (proj1 Hty).
Qed.

Theorem type_simple_in_term_satisfied
    : type_simple_in_term_satisfied_statement.
Proof.
  intros exempt statics valuation tenv min_name_mode s canonical ty
    Hsatisfied.
  unfold Typing_env.type_simple_in_term_exn.
  (* The value of the [Simple], if it has one, is of the kind of the type
     that the environment finds. *)
  assert (Hfound :
    ∀ (ty' : flambda_type)
      (binding_time_and_mode : Binding_time.With_name_mode.t) (v : value),
      match s with
      | Const const =>
        Some (type_for_const const, Binding_time.With_name_mode.consts)
      | Var x =>
        Typing_env.find_with_binding_time_and_mode tenv (Name_var x) None
      | Symbol sym =>
        Typing_env.find_with_binding_time_and_mode tenv (Name_symbol sym) None
      end = Some (ty', binding_time_and_mode) ->
      eval_simple valuation s = Some v ->
      kind_of_value v = FlambdaTypeGrammar.kind ty').
  { intros ty' binding_time_and_mode v. destruct s as [x | sym | c].
    - intros Hfind Hvalue.
      exact (has_type_unless_kind _ statics valuation v ty'
               (find_with_binding_time_and_mode_satisfied exempt statics
                  valuation tenv (Name_var x) None ty' binding_time_and_mode
                  v Hsatisfied Hfind Hvalue)).
    - intros Hfind Hvalue.
      exact (has_type_unless_kind _ statics valuation v ty'
               (find_with_binding_time_and_mode_satisfied exempt statics
                  valuation tenv (Name_symbol sym) None ty'
                  binding_time_and_mode v Hsatisfied Hfind Hvalue)).
    - intros [= <- _] [= <-]. rewrite kind_type_for_const. reflexivity. }
  destruct (match s with
            | Const const =>
              Some (type_for_const const, Binding_time.With_name_mode.consts)
            | Var x =>
              Typing_env.find_with_binding_time_and_mode tenv (Name_var x) None
            | Symbol sym =>
              Typing_env.find_with_binding_time_and_mode tenv
                (Name_symbol sym) None
            end) as [[ty' binding_time_and_mode] | ]; [ | discriminate ].
  destruct (Aliases.get_canonical_element_exn _ _ _ _ _ _ _) as [alias | ]
    eqn:Hcanonical; [ | discriminate ].
  intros [= <- <-].
  destruct Hsatisfied as [_ [_ Hrespected]].
  assert (Halias : eval_simple valuation alias = eval_simple valuation s)
    by exact (respected_canonical_at_mode _ _ _ _ valuation _ _ _ _ _
                Hrespected Hcanonical).
  split; [ exact Halias | ].
  intros v Hvalue. apply alias_satisfied.
  split; [ congruence | ].
  exact (Hfound ty' binding_time_and_mode v eq_refl Hvalue).
Qed.

Theorem type_simple_in_term_canonical
    : type_simple_in_term_canonical_statement.
Proof.
  intros tenv statics vars s canonical ty
    [_ [valuation [Hsatisfied Hagrees]]] Hmem Hcanonical.
  (* The [Simple] and its canonical element have, in the machine, the
     values that the valuation gives them. *)
  assert (Hs : eval_simple vars s = eval_simple valuation s).
  { destruct s as [x | sym | c]; [ | reflexivity | reflexivity ].
    symmetry. exact (Hagrees x Hmem). }
  assert (Hc : eval_simple vars canonical = eval_simple valuation canonical).
  { destruct canonical as [y | sym | c]; [ | reflexivity | reflexivity ].
    revert Hcanonical. unfold Typing_env.type_simple_in_term_exn.
    destruct (match s with
              | Const const =>
                Some (type_for_const const, Binding_time.With_name_mode.consts)
              | Var x =>
                Typing_env.find_with_binding_time_and_mode tenv (Name_var x)
                  None
              | Symbol sym =>
                Typing_env.find_with_binding_time_and_mode tenv
                  (Name_symbol sym) None
              end) as [[ty' binding_time_and_mode] | ]; [ | discriminate ].
    destruct (Aliases.get_canonical_element_exn _ _ _ _ _ _ _) as [alias | ]
      eqn:Halias; [ | discriminate ].
    intros [= _ ->].
    apply get_canonical_element_exn_normal in Halias.
    destruct Halias as [-> | [data [binding_time_and_mode' [Hlookup Hmode]]]].
    - exact Hs.
    - symmetry. apply Hagrees. apply in_scope_in_terms_lookup.
      exists data, binding_time_and_mode'. split; assumption. }
  rewrite Hs, Hc.
  exact (proj1 (type_simple_in_term_satisfied _ statics valuation tenv
                  (Some Name_mode.normal) s canonical ty Hsatisfied
                  Hcanonical)).
Qed.
