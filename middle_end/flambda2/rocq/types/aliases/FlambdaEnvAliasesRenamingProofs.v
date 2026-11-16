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

(* Proofs of the statements of FlambdaEnvAliasesRenamingSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaNameAbstractionSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeRenameSpec
  FlambdaTypeRenameProofs FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaEnvAliasesRenamingSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Lists and maps whose keys are renamed                                    *)
(* ======================================================================== *)

Lemma filter_renamed {B C : Type} (g : B -> C) (p : B -> bool)
    (p' : C -> bool) (l : list B) :
  (∀ b : B, p' (g b) = p b) -> filter p' (map g l) = map g (filter p l).
Proof.
  intros Hp. induction l as [ | b l IH]; simpl; [ reflexivity | ].
  rewrite Hp. destruct (p b); simpl; rewrite IH; reflexivity.
Qed.

Lemma is_empty_map {B C : Type} (g : B -> C) (l : list B) :
  Keyed.is_empty (map g l) = Keyed.is_empty l.
Proof. destruct l; reflexivity. Qed.

Lemma forallb_renamed {B C : Type} (g : B -> C) (p : B -> bool)
    (p' : C -> bool) (l : list B) :
  (∀ b : B, p' (g b) = p b) -> forallb p' (map g l) = forallb p l.
Proof.
  intros Hp. induction l as [ | b l IH]; simpl; [ reflexivity | ].
  rewrite Hp, IH. reflexivity.
Qed.

Lemma fold_left_renamed {B B' C C' : Type} (gb : B -> B') (gc : C -> C')
    (f : B -> C -> B) (f' : B' -> C' -> B') (l : list C) :
  (∀ (b : B) (c : C), f' (gb b) (gc c) = gb (f b c)) ->
  ∀ init : B,
    List.fold_left f' (map gc l) (gb init) = gb (List.fold_left f l init).
Proof.
  intros Hf. induction l as [ | c l IH]; intros init; simpl;
    [ reflexivity | ].
  rewrite Hf. apply IH.
Qed.

(* The earliest of a list, as [Aliases.find_earliest_alias] finds it. *)
Lemma fold_earliest_renamed {B : Type} (g : B -> B)
    (earlier earlier' : B -> B -> bool) (l : list B) (first : B) :
  (∀ a b : B, earlier' (g a) (g b) = earlier a b) ->
  List.fold_left
    (fun min_elt elt => if earlier' elt min_elt then elt else min_elt)
    (map g l) (g first)
  = g (List.fold_left
         (fun min_elt elt => if earlier elt min_elt then elt else min_elt) l
         first).
Proof.
  intros Hearlier. revert first.
  induction l as [ | elt l IH]; intros first; simpl; [ reflexivity | ].
  rewrite Hearlier. destruct (earlier elt first); apply IH.
Qed.

Lemma map_bindings_identity {K B : Type} (f : K -> K) (g : B -> B)
    (bindings : Keyed.finite_map K B) :
  (∀ k : K, f k = k) -> (∀ b : B, g b = b) ->
  Keyed.map_bindings f g bindings = bindings.
Proof.
  intros Hf Hg. unfold Keyed.map_bindings.
  induction bindings as [ | [k b] bindings IH]; simpl; [ reflexivity | ].
  rewrite Hf, Hg, IH. reflexivity.
Qed.

Lemma map_bindings_compose {K B C D : Type} (f f' : K -> K) (g : C -> D)
    (g' : B -> C) (bindings : Keyed.finite_map K B) :
  Keyed.map_bindings f g (Keyed.map_bindings f' g' bindings)
  = Keyed.map_bindings (fun k => f (f' k)) (fun b => g (g' b)) bindings.
Proof. unfold Keyed.map_bindings. rewrite map_map. reflexivity. Qed.

Lemma map_bindings_extensional {K B C : Type} (f f' : K -> K)
    (g g' : B -> C) (bindings : Keyed.finite_map K B) :
  (∀ k : K, f k = f' k) -> (∀ b : B, g b = g' b) ->
  Keyed.map_bindings f g bindings = Keyed.map_bindings f' g' bindings.
Proof.
  intros Hf Hg. apply map_ext. intros [k b]. simpl. rewrite Hf, Hg.
  reflexivity.
Qed.

Lemma keys_renamed {K B C : Type} (f : K -> K) (g : B -> C)
    (bindings : Keyed.finite_map K B) :
  map fst (Keyed.map_bindings f g bindings) = map f (map fst bindings).
Proof. unfold Keyed.map_bindings. rewrite !map_map. reflexivity. Qed.

Section Renamed_keys.

(* A renaming of keys that the equality test of keys does not see. *)
Context {K : Type} (eqb : K -> K -> bool) (f : K -> K).
Hypothesis eqb_renamed : ∀ k k' : K, eqb (f k) (f k') = eqb k k'.

Lemma among_keys_renamed (k : K) (keys : list K) :
  Keyed.among eqb (f k) (map f keys) = Keyed.among eqb k keys.
Proof.
  unfold Keyed.among.
  induction keys as [ | k' keys IH]; simpl; [ reflexivity | ].
  rewrite eqb_renamed, IH. reflexivity.
Qed.

Lemma without_renamed (removed keys : list K) :
  Keyed.without eqb (map f removed) (map f keys)
  = map f (Keyed.without eqb removed keys).
Proof.
  unfold Keyed.without. apply filter_renamed. intros k.
  rewrite among_keys_renamed. reflexivity.
Qed.

Lemma inter_keys_renamed (keys keys' : list K) :
  Keyed.inter eqb (map f keys) (map f keys')
  = map f (Keyed.inter eqb keys keys').
Proof.
  unfold Keyed.inter. apply filter_renamed. intros k.
  apply among_keys_renamed.
Qed.

Lemma dedup_renamed (keys : list K) :
  Keyed.dedup eqb (map f keys) = map f (Keyed.dedup eqb keys).
Proof.
  induction keys as [ | k keys IH]; simpl; [ reflexivity | ].
  rewrite IH. f_equal. apply (without_renamed [k]).
Qed.

Lemma lookup_renamed {B C : Type} (g : B -> C) (k : K)
    (bindings : Keyed.finite_map K B) :
  Keyed.lookup eqb (f k) (Keyed.map_bindings f g bindings)
  = option_map g (Keyed.lookup eqb k bindings).
Proof.
  unfold Keyed.map_bindings.
  induction bindings as [ | [k' b] bindings IH]; simpl; [ reflexivity | ].
  rewrite eqb_renamed. destruct (eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma is_bound_renamed {B C : Type} (g : B -> C) (k : K)
    (bindings : Keyed.finite_map K B) :
  Keyed.is_bound eqb (f k) (Keyed.map_bindings f g bindings)
  = Keyed.is_bound eqb k bindings.
Proof.
  unfold Keyed.is_bound. rewrite lookup_renamed.
  destruct (Keyed.lookup eqb k bindings); reflexivity.
Qed.

Lemma domain_renamed {B C : Type} (g : B -> C)
    (bindings : Keyed.finite_map K B) :
  Keyed.domain eqb (Keyed.map_bindings f g bindings)
  = map f (Keyed.domain eqb bindings).
Proof. unfold Keyed.domain. rewrite keys_renamed. apply dedup_renamed. Qed.

Lemma tabulate_renamed {B C : Type} (g : B -> C) (bound : K -> option B)
    (bound' : K -> option C) (keys : list K) :
  (∀ k : K, bound' (f k) = option_map g (bound k)) ->
  Keyed.tabulate bound' (map f keys)
  = Keyed.map_bindings f g (Keyed.tabulate bound keys).
Proof.
  intros Hbound. unfold Keyed.tabulate, Keyed.map_bindings.
  induction keys as [ | k keys IH]; simpl; [ reflexivity | ].
  rewrite Hbound, IH, map_app. destruct (bound k); reflexivity.
Qed.

Lemma compact_keys_renamed {B C : Type} (g : B -> C)
    (bindings : Keyed.finite_map K B) :
  Keyed.compact eqb (Keyed.map_bindings f g bindings)
  = Keyed.map_bindings f g (Keyed.compact eqb bindings).
Proof.
  unfold Keyed.compact. rewrite domain_renamed. apply tabulate_renamed.
  intros k. apply lookup_renamed.
Qed.

Lemma filter_bindings_renamed {B C : Type} (g : B -> C)
    (keep : K -> B -> bool) (keep' : K -> C -> bool)
    (bindings : Keyed.finite_map K B) :
  (∀ (k : K) (b : B), keep' (f k) (g b) = keep k b) ->
  Keyed.filter_bindings eqb keep' (Keyed.map_bindings f g bindings)
  = Keyed.map_bindings f g (Keyed.filter_bindings eqb keep bindings).
Proof.
  intros Hkeep. unfold Keyed.filter_bindings. rewrite domain_renamed.
  apply tabulate_renamed. intros k. rewrite lookup_renamed.
  destruct (Keyed.lookup eqb k bindings) as [b | ]; simpl; [ | reflexivity ].
  rewrite Hkeep. destruct (keep k b); reflexivity.
Qed.

Lemma unbind_renamed {B C : Type} (g : B -> C) (k : K)
    (bindings : Keyed.finite_map K B) :
  Keyed.unbind eqb (f k) (Keyed.map_bindings f g bindings)
  = Keyed.map_bindings f g (Keyed.unbind eqb k bindings).
Proof.
  unfold Keyed.unbind, Keyed.map_bindings. apply filter_renamed.
  intros [k' b]. simpl. rewrite eqb_renamed. reflexivity.
Qed.

End Renamed_keys.

(* ======================================================================== *)
(* Names and simples                                                        *)
(* ======================================================================== *)

Lemma rename_name_compose (second first : renaming) (n : name) :
  rename_name (compose_renaming second first) n
  = rename_name second (rename_name first n).
Proof.
  destruct n; simpl; [ rewrite rename_variable_compose | ]; reflexivity.
Qed.

Lemma rename_name_inverse_l (r : renaming) (n : name) :
  rename_name (inverse_renaming r) (rename_name r n) = n.
Proof.
  destruct n; simpl; [ rewrite rename_variable_inverse_l | ]; reflexivity.
Qed.

Lemma rename_simple_inverse_l (r : renaming) (s : simple) :
  rename_simple (inverse_renaming r) (rename_simple r s) = s.
Proof.
  destruct s; simpl; [ rewrite rename_variable_inverse_l | | ]; reflexivity.
Qed.

Lemma rename_simple_inverse_r (r : renaming) (s : simple) :
  rename_simple r (rename_simple (inverse_renaming r) s) = s.
Proof.
  destruct s; simpl; [ rewrite rename_variable_inverse_r | | ]; reflexivity.
Qed.

Lemma rename_simple_of_name (r : renaming) (n : name) :
  rename_simple r (simple_of_name n) = simple_of_name (rename_name r n).
Proof. destruct n; reflexivity. Qed.

Lemma name_of_simple_renamed (r : renaming) (s : simple) :
  name_of_simple (rename_simple r s)
  = option_map (rename_name r) (name_of_simple s).
Proof. destruct s; reflexivity. Qed.

Lemma simple_equal_renamed (r : renaming) (s s' : simple) :
  simple_equal (rename_simple r s) (rename_simple r s') = simple_equal s s'.
Proof. destruct s, s'; simpl; try reflexivity. apply rename_variable_eqb. Qed.

Lemma name_is_symbol_renamed (r : renaming) (n : name) :
  name_is_symbol (rename_name r n) = name_is_symbol n.
Proof. destruct n; reflexivity. Qed.

Lemma eval_simple_rename_env (r : renaming) (vars : env) (s : simple) :
  eval_simple (rename_env r vars) (rename_simple r s) = eval_simple vars s.
Proof. destruct s; simpl; [ apply lookup_rename_env | | ]; reflexivity. Qed.

Lemma rename_env_inverse_l (r : renaming) (vars : env) :
  rename_env (inverse_renaming r) (rename_env r vars) = vars.
Proof.
  unfold rename_env. rewrite map_map.
  induction vars as [ | [x v] vars IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_inverse_l, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* Binding times and the order                                              *)
(* ======================================================================== *)

Lemma rename_resolver_renamed (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (n : name) :
  rename_resolver r binding_time_resolver (rename_name r n)
  = binding_time_resolver n.
Proof. unfold rename_resolver. rewrite rename_name_inverse_l. reflexivity. Qed.

Lemma lookup_binding_time_renamed {A : Type} (r : renaming)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (n : name) :
  Keyed.lookup name_eqb (rename_name r n)
    (rename_binding_times r binding_times_and_modes)
  = Keyed.lookup name_eqb n binding_times_and_modes.
Proof.
  unfold rename_binding_times.
  rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  destruct (Keyed.lookup name_eqb n binding_times_and_modes); reflexivity.
Qed.

Lemma name_defined_earlier_renamed {A : Type} (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (n n' : name) :
  Aliases.name_defined_earlier (rename_resolver r binding_time_resolver)
    (rename_binding_times r binding_times_and_modes) (rename_name r n)
    (rename_name r n')
  = Aliases.name_defined_earlier binding_time_resolver binding_times_and_modes
      n n'.
Proof.
  unfold Aliases.name_defined_earlier.
  rewrite name_eqb_renamed, !name_is_symbol_renamed,
    !lookup_binding_time_renamed, !rename_resolver_renamed.
  reflexivity.
Qed.

Theorem rename_defined_earlier : rename_defined_earlier_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes e e'.
  unfold Aliases.defined_earlier. rewrite !name_of_simple_renamed.
  destruct (name_of_simple e') as [n' | ]; simpl; [ | reflexivity ].
  destruct (name_of_simple e) as [n | ]; simpl; [ | reflexivity ].
  apply name_defined_earlier_renamed.
Qed.

Lemma binding_time_and_name_mode_renamed {A : Type} (r : renaming)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (e : simple) :
  Aliases.binding_time_and_name_mode
    (rename_binding_times r binding_times_and_modes) (rename_simple r e)
  = Aliases.binding_time_and_name_mode binding_times_and_modes e.
Proof.
  destruct e as [x | s | c]; try reflexivity.
  unfold Aliases.binding_time_and_name_mode, rename_simple.
  change (Name_var (rename_variable r x)) with (rename_name r (Name_var x)).
  rewrite lookup_binding_time_renamed. reflexivity.
Qed.

Lemma compute_name_mode_unscoped_renamed {A : Type} (r : renaming)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (e : simple) :
  Aliases.compute_name_mode_unscoped
    (rename_binding_times r binding_times_and_modes) (rename_simple r e)
  = Aliases.compute_name_mode_unscoped binding_times_and_modes e.
Proof.
  unfold Aliases.compute_name_mode_unscoped.
  rewrite binding_time_and_name_mode_renamed. reflexivity.
Qed.

Lemma compute_name_mode_renamed {A : Type} (r : renaming)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (e : simple) (min_binding_time : Binding_time.t) :
  Aliases.compute_name_mode (rename_binding_times r binding_times_and_modes)
    (rename_simple r e) min_binding_time
  = Aliases.compute_name_mode binding_times_and_modes e min_binding_time.
Proof.
  unfold Aliases.compute_name_mode.
  rewrite binding_time_and_name_mode_renamed. reflexivity.
Qed.

(* ======================================================================== *)
(* The aliases of one canonical element                                     *)
(* ======================================================================== *)

Lemma all_renamed (r : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  Aliases.Aliases_of_canonical_element.all
    (rename_aliases_of_canonical_element r aliases)
  = map (rename_name r) (Aliases.Aliases_of_canonical_element.all aliases).
Proof. apply domain_renamed. exact (name_eqb_renamed r). Qed.

Lemma mem_renamed (r : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t) (n : name) :
  Aliases.Aliases_of_canonical_element.mem
    (rename_aliases_of_canonical_element r aliases) (rename_name r n)
  = Aliases.Aliases_of_canonical_element.mem aliases n.
Proof. apply is_bound_renamed. exact (name_eqb_renamed r). Qed.

Lemma add_alias_renamed (r : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t) (n : name)
    (name_mode : Name_mode.t) :
  Aliases.Aliases_of_canonical_element.add
    (rename_aliases_of_canonical_element r aliases) (rename_name r n)
    name_mode
  = option_map (rename_aliases_of_canonical_element r)
      (Aliases.Aliases_of_canonical_element.add aliases n name_mode).
Proof.
  unfold Aliases.Aliases_of_canonical_element.add. rewrite mem_renamed.
  destruct (Aliases.Aliases_of_canonical_element.mem aliases n); reflexivity.
Qed.

Lemma union_renamed (r : renaming)
    (aliases aliases' : Aliases.Aliases_of_canonical_element.t) :
  Aliases.Aliases_of_canonical_element.union
    (rename_aliases_of_canonical_element r aliases)
    (rename_aliases_of_canonical_element r aliases')
  = rename_aliases_of_canonical_element r
      (Aliases.Aliases_of_canonical_element.union aliases aliases').
Proof. symmetry. apply map_app. Qed.

Lemma at_mode_renamed (r : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t) (order : Name_mode.t) :
  Aliases.Aliases_of_canonical_element.at_mode
    (rename_aliases_of_canonical_element r aliases) order
  = map (rename_name r)
      (Aliases.Aliases_of_canonical_element.at_mode aliases order).
Proof.
  unfold Aliases.Aliases_of_canonical_element.at_mode,
    rename_aliases_of_canonical_element.
  rewrite
    (filter_bindings_renamed name_eqb (rename_name r) (name_eqb_renamed r)
       (fun name_mode : Name_mode.t => name_mode)
       (fun _ name_mode => Name_mode.equal name_mode order)
       (fun _ name_mode => Name_mode.equal name_mode order)).
  - apply keys_renamed.
  - reflexivity.
Qed.

Lemma find_earliest_candidates_renamed (r : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t)
    (filter_by_scope filter_by_scope' : Name_mode.t -> list name -> list name)
    (min_name_mode : Name_mode.t) :
  (∀ (order : Name_mode.t) (names : list name),
    filter_by_scope' order (map (rename_name r) names)
    = map (rename_name r) (filter_by_scope order names)) ->
  Aliases.Aliases_of_canonical_element.find_earliest_candidates
    (rename_aliases_of_canonical_element r aliases) filter_by_scope'
    min_name_mode
  = option_map (map (rename_name r))
      (Aliases.Aliases_of_canonical_element.find_earliest_candidates aliases
         filter_by_scope min_name_mode).
Proof.
  intros Hscope.
  unfold Aliases.Aliases_of_canonical_element.find_earliest_candidates.
  cbn [List.fold_left]. rewrite !at_mode_renamed, !Hscope, !is_empty_map.
  destruct min_name_mode; cbn;
    repeat
      match goal with
      | |- context [Keyed.is_empty ?names] =>
        destruct names; cbn [Keyed.is_empty map option_map]
      end;
    reflexivity.
Qed.

(* ======================================================================== *)
(* The structure                                                            *)
(* ======================================================================== *)

Lemma rename_aliases_of_canonical_element_identity
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  rename_aliases_of_canonical_element identity_renaming aliases = aliases.
Proof.
  apply map_bindings_identity;
    [ apply rename_name_identity | reflexivity ].
Qed.

Theorem rename_aliases_identity : rename_aliases_identity_statement.
Proof.
  intros [canonical_elements of_names of_consts]. unfold rename_aliases.
  simpl. f_equal.
  - apply map_bindings_identity;
      [ apply rename_name_identity | apply rename_simple_id ].
  - apply map_bindings_identity;
      [ apply rename_name_identity
      | apply rename_aliases_of_canonical_element_identity ].
  - apply (map_bindings_identity (fun c : const => c));
      [ reflexivity | apply rename_aliases_of_canonical_element_identity ].
Qed.

Lemma rename_aliases_of_canonical_element_extensional (r r' : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  (∀ x : variable, rename_variable r x = rename_variable r' x) ->
  rename_aliases_of_canonical_element r aliases
  = rename_aliases_of_canonical_element r' aliases.
Proof.
  intros Hsame. apply map_bindings_extensional; [ | reflexivity ].
  intros [x | s]; simpl; [ rewrite Hsame | ]; reflexivity.
Qed.

Theorem rename_aliases_extensional : rename_aliases_extensional_statement.
Proof.
  intros r r' [canonical_elements of_names of_consts] Hsame.
  assert (Hname : ∀ n : name, rename_name r n = rename_name r' n).
  { intros [x | s]; simpl; [ rewrite Hsame | ]; reflexivity. }
  unfold rename_aliases. simpl. f_equal.
  - apply map_bindings_extensional; [ exact Hname | ].
    intros [x | s | c]; simpl; [ rewrite Hsame | | ]; reflexivity.
  - apply map_bindings_extensional; [ exact Hname | ].
    intros aliases.
    apply rename_aliases_of_canonical_element_extensional, Hsame.
  - apply (map_bindings_extensional (fun c : const => c)
             (fun c : const => c));
      [ reflexivity | ].
    intros aliases.
    apply rename_aliases_of_canonical_element_extensional, Hsame.
Qed.

Lemma rename_aliases_of_canonical_element_compose (second first : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  rename_aliases_of_canonical_element (compose_renaming second first) aliases
  = rename_aliases_of_canonical_element second
      (rename_aliases_of_canonical_element first aliases).
Proof.
  unfold rename_aliases_of_canonical_element.
  rewrite map_bindings_compose.
  apply map_bindings_extensional;
    [ apply rename_name_compose | reflexivity ].
Qed.

Theorem rename_aliases_compose : rename_aliases_compose_statement.
Proof.
  intros second first [canonical_elements of_names of_consts].
  unfold rename_aliases. simpl. f_equal.
  - rewrite map_bindings_compose.
    apply map_bindings_extensional;
      [ apply rename_name_compose | apply rename_simple_compose ].
  - rewrite map_bindings_compose.
    apply map_bindings_extensional;
      [ apply rename_name_compose
      | apply rename_aliases_of_canonical_element_compose ].
  - unfold Keyed.map_values. rewrite map_map. apply map_ext.
    intros [c aliases]. simpl.
    rewrite rename_aliases_of_canonical_element_compose. reflexivity.
Qed.

Lemma rename_aliases_inverse_l (r : renaming) (t : Aliases.t) :
  rename_aliases (inverse_renaming r) (rename_aliases r t) = t.
Proof.
  rewrite <- rename_aliases_compose.
  rewrite (rename_aliases_extensional _ identity_renaming).
  - apply rename_aliases_identity.
  - intros x. rewrite rename_variable_compose, rename_variable_inverse_l.
    reflexivity.
Qed.

(* What [Aliases.canonical] answers, renamed. *)
Definition rename_is_canonical_or_alias (r : renaming)
    (canonical : Aliases.is_canonical_or_alias)
    : Aliases.is_canonical_or_alias :=
  match canonical with
  | Aliases.Is_canonical => Aliases.Is_canonical
  | Aliases.Alias_of_canonical canonical_element =>
    Aliases.Alias_of_canonical (rename_simple r canonical_element)
  end.

Lemma canonical_renamed (r : renaming) (t : Aliases.t) (e : simple) :
  Aliases.canonical (rename_aliases r t) (rename_simple r e)
  = rename_is_canonical_or_alias r (Aliases.canonical t e).
Proof.
  unfold Aliases.canonical. rewrite name_of_simple_renamed.
  destruct (name_of_simple e) as [n | ]; simpl; [ | reflexivity ].
  rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  destruct (Keyed.lookup name_eqb n (Aliases.canonical_elements t));
    reflexivity.
Qed.

Lemma demoted_to_renamed (r : renaming) (t : Aliases.t) (n : name)
    (canonical_element : simple) :
  demoted_to t n canonical_element ->
  demoted_to (rename_aliases r t) (rename_name r n)
    (rename_simple r canonical_element).
Proof.
  unfold demoted_to. intros Hdemoted. simpl.
  rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r)),
    Hdemoted.
  reflexivity.
Qed.

Lemma get_aliases_of_canonical_element_renamed (r : renaming) (t : Aliases.t)
    (e : simple) :
  Aliases.get_aliases_of_canonical_element (rename_aliases r t)
    (rename_simple r e)
  = rename_aliases_of_canonical_element r
      (Aliases.get_aliases_of_canonical_element t e).
Proof.
  unfold Aliases.get_aliases_of_canonical_element.
  destruct e as [x | s | c]; simpl.
  - change (Name_var (rename_variable r x)) with (rename_name r (Name_var x)).
    rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
    destruct
      (Keyed.lookup name_eqb (Name_var x)
         (Aliases.aliases_of_canonical_names t));
      reflexivity.
  - change (Name_symbol s) with (rename_name r (Name_symbol s)) at 1.
    rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
    destruct
      (Keyed.lookup name_eqb (Name_symbol s)
         (Aliases.aliases_of_canonical_names t));
      reflexivity.
  - rewrite keyed_lookup_map_values.
    destruct (Keyed.lookup const_eqb c (Aliases.aliases_of_consts t));
      reflexivity.
Qed.

Lemma listed_alias_renamed (r : renaming) (t : Aliases.t)
    (canonical_element : simple) (n : name) :
  listed_alias (rename_aliases r t) (rename_simple r canonical_element)
    (rename_name r n)
  <-> listed_alias t canonical_element n.
Proof.
  unfold listed_alias.
  rewrite get_aliases_of_canonical_element_renamed, mem_renamed. reflexivity.
Qed.

Lemma mentioned_renamed (r : renaming) (t : Aliases.t) (n : name) :
  aliases_mention t n ->
  aliases_mention (rename_aliases r t) (rename_name r n).
Proof.
  intros [[canonical_element Hdemoted] | [[n' Hdemoted] | [[canonical_element
    Hlisted] | [n' Hlisted]]]].
  - left. exists (rename_simple r canonical_element).
    apply demoted_to_renamed, Hdemoted.
  - right. left. exists (rename_name r n').
    rewrite <- rename_simple_of_name. apply demoted_to_renamed, Hdemoted.
  - right. right. left. exists (rename_simple r canonical_element).
    apply listed_alias_renamed, Hlisted.
  - right. right. right. exists (rename_name r n').
    rewrite <- rename_simple_of_name. apply listed_alias_renamed, Hlisted.
Qed.

Theorem rename_mentioned : rename_mentioned_statement.
Proof.
  intros r t n. split; [ | apply mentioned_renamed ].
  intros Hmentioned.
  apply (mentioned_renamed (inverse_renaming r)) in Hmentioned.
  rewrite rename_aliases_inverse_l, rename_name_inverse_l in Hmentioned.
  exact Hmentioned.
Qed.

(* ======================================================================== *)
(* The queries                                                              *)
(* ======================================================================== *)

Theorem rename_canonical : rename_canonical_statement.
Proof.
  intros r t n. unfold Aliases.get_canonical_ignoring_name_mode.
  rewrite <- rename_simple_of_name, canonical_renamed.
  destruct (Aliases.canonical t (simple_of_name n)); reflexivity.
Qed.

Lemma find_earliest_alias_renamed {A : Type} (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element : simple)
    (min_binding_time : Binding_time.t) (min_name_mode : Name_mode.t) :
  Aliases.find_earliest_alias (rename_resolver r binding_time_resolver)
    (rename_binding_times r binding_times_and_modes) (rename_aliases r t)
    (rename_simple r canonical_element) min_binding_time min_name_mode
  = option_map (rename_simple r)
      (Aliases.find_earliest_alias binding_time_resolver
         binding_times_and_modes t canonical_element min_binding_time
         min_name_mode).
Proof.
  unfold Aliases.find_earliest_alias. cbv zeta.
  rewrite get_aliases_of_canonical_element_renamed.
  rewrite
    (find_earliest_candidates_renamed r _
       (fun name_mode names =>
          if Name_mode.equal name_mode Name_mode.in_types
          then names
          else
            filter
              (fun name =>
                 Name_mode.equal name_mode
                   (Aliases.compute_name_mode binding_times_and_modes
                      (simple_of_name name) min_binding_time))
              names)).
  2: {
    intros order names.
    destruct (Name_mode.equal order Name_mode.in_types); [ reflexivity | ].
    apply filter_renamed. intros n.
    rewrite <- rename_simple_of_name, compute_name_mode_renamed.
    reflexivity. }
  match goal with
  | |- context [option_map (map (rename_name r)) ?candidates] =>
    destruct candidates as [[ | first names] | ]
  end; try reflexivity.
  cbn [option_map].
  change (map (rename_name r) (first :: names))
    with (rename_name r first :: map (rename_name r) names).
  cbv iota.
  change (rename_name r first :: map (rename_name r) names)
    with (map (rename_name r) (first :: names)).
  rewrite
    (fold_earliest_renamed (rename_name r)
       (Aliases.name_defined_earlier binding_time_resolver
          binding_times_and_modes)).
  - rewrite rename_simple_of_name. reflexivity.
  - intros n n'. apply name_defined_earlier_renamed.
Qed.

Theorem rename_canonical_at_mode : rename_canonical_at_mode_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes t element
    elt_name_mode min_name_mode min_binding_time.
  unfold Aliases.get_canonical_element_exn. cbv zeta.
  rewrite canonical_renamed.
  destruct (Aliases.canonical t element) as [ | canonical_element];
    cbn [rename_is_canonical_or_alias].
  - destruct
      (match Name_mode.compare_partial_order elt_name_mode min_name_mode with
       | Some c => 0 <=? c
       | None => false
       end);
      [ reflexivity | apply find_earliest_alias_renamed ].
  - rewrite compute_name_mode_renamed.
    assert
      (Hsame
        : ∀ (b : bool) (found found' : option simple),
            found' = option_map (rename_simple r) found ->
            (if b then Some (rename_simple r canonical_element) else found')
            = option_map (rename_simple r)
                (if b then Some canonical_element else found)).
    { intros [ | ] found found' ->; reflexivity. }
    destruct
      (match Name_mode.compare_partial_order elt_name_mode min_name_mode with
       | Some c => 0 <=? c
       | None => false
       end);
      apply Hsame, find_earliest_alias_renamed.
Qed.

(* ======================================================================== *)
(* Adding an alias                                                          *)
(* ======================================================================== *)

Lemma add_alias_between_canonical_elements_renamed {A : Type} (r : renaming)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element to_be_demoted : simple) :
  Aliases.add_alias_between_canonical_elements
    (rename_binding_times r binding_times_and_modes) (rename_aliases r t)
    (rename_simple r canonical_element) (rename_simple r to_be_demoted)
  = match
      Aliases.add_alias_between_canonical_elements binding_times_and_modes t
        canonical_element to_be_demoted
    with
    | None => None
    | Some Or_bottom.Bottom => Some Or_bottom.Bottom
    | Some (Or_bottom.Ok t') => Some (Or_bottom.Ok (rename_aliases r t'))
    end.
Proof.
  unfold Aliases.add_alias_between_canonical_elements.
  rewrite simple_equal_renamed.
  destruct (simple_equal canonical_element to_be_demoted); [ reflexivity | ].
  rewrite name_of_simple_renamed.
  destruct (name_of_simple to_be_demoted) as [name_to_be_demoted | ];
    [ | reflexivity ].
  cbn [option_map]. cbv zeta.
  rewrite !get_aliases_of_canonical_element_renamed, union_renamed,
    add_alias_renamed, compute_name_mode_unscoped_renamed.
  destruct
    (Aliases.Aliases_of_canonical_element.add
       (Aliases.Aliases_of_canonical_element.union
          (Aliases.get_aliases_of_canonical_element t to_be_demoted)
          (Aliases.get_aliases_of_canonical_element t canonical_element))
       name_to_be_demoted
       (Aliases.compute_name_mode_unscoped binding_times_and_modes
          to_be_demoted))
    as [aliases | ];
    [ | reflexivity ].
  cbn [option_map].
  assert
    (Hcanonical_elements
      : (rename_name r name_to_be_demoted, rename_simple r canonical_element)
        :: map (fun alias => (alias, rename_simple r canonical_element))
             (Aliases.Aliases_of_canonical_element.all
                (rename_aliases_of_canonical_element r
                   (Aliases.get_aliases_of_canonical_element t
                      to_be_demoted)))
           ++ Aliases.canonical_elements (rename_aliases r t)
        = Keyed.map_bindings (rename_name r) (rename_simple r)
            ((name_to_be_demoted, canonical_element)
             :: map (fun alias => (alias, canonical_element))
                  (Aliases.Aliases_of_canonical_element.all
                     (Aliases.get_aliases_of_canonical_element t
                        to_be_demoted))
                ++ Aliases.canonical_elements t)).
  { rewrite all_renamed. unfold Keyed.map_bindings. simpl.
    rewrite map_app, !map_map. reflexivity. }
  assert
    (Hunbound
      : Keyed.unbind name_eqb (rename_name r name_to_be_demoted)
          (Aliases.aliases_of_canonical_names (rename_aliases r t))
        = Keyed.map_bindings (rename_name r)
            (rename_aliases_of_canonical_element r)
            (Keyed.unbind name_eqb name_to_be_demoted
               (Aliases.aliases_of_canonical_names t))).
  { apply (unbind_renamed name_eqb (rename_name r) (name_eqb_renamed r)). }
  rewrite Hcanonical_elements, Hunbound.
  destruct canonical_element as [x | s | c]; reflexivity.
Qed.

Theorem rename_add : rename_add_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2.
  unfold Aliases.add, Aliases.add0, Aliases.add_alias,
    Aliases.choose_canonical_element_to_be_demoted.
  rewrite rename_defined_earlier.
  destruct
    (Aliases.defined_earlier binding_time_resolver binding_times_and_modes
       canonical_element1 canonical_element2);
    cbv zeta; cbv iota; cbv beta;
    rewrite add_alias_between_canonical_elements_renamed;
    match goal with
    | |- context [Aliases.add_alias_between_canonical_elements ?b ?t ?e ?e'] =>
      destruct (Aliases.add_alias_between_canonical_elements b t e e')
        as [[t' | ] | ]
    end;
    try reflexivity; simpl; rewrite name_of_simple_renamed;
    match goal with
    | |- context [name_of_simple ?e] => destruct (name_of_simple e)
    end;
    reflexivity.
Qed.

(* ======================================================================== *)
(* Respect                                                                  *)
(* ======================================================================== *)

Lemma value_of_name_rename_env (r : renaming) (vars : env) (n : name) :
  value_of_name (rename_env r vars) (rename_name r n) = value_of_name vars n.
Proof.
  unfold value_of_name.
  rewrite <- rename_simple_of_name. apply eval_simple_rename_env.
Qed.

Lemma respected_renamed (r : renaming) (t : Aliases.t) (vars : env) :
  aliases_respected t vars ->
  aliases_respected (rename_aliases r t) (rename_env r vars).
Proof.
  intros [Hcanonical Hlisted]. split.
  - intros n canonical_element Hdemoted.
    rewrite <- (rename_name_inverse_r r n) in Hdemoted |- *.
    unfold demoted_to in Hdemoted. simpl in Hdemoted.
    rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r))
      in Hdemoted.
    destruct
      (Keyed.lookup name_eqb (rename_name (inverse_renaming r) n)
         (Aliases.canonical_elements t))
      as [canonical_element' | ] eqn:Hlookup;
      [ | discriminate ].
    injection Hdemoted as <-.
    rewrite value_of_name_rename_env, eval_simple_rename_env.
    apply Hcanonical, Hlookup.
  - intros canonical_element n Hlisted'.
    rewrite <- (rename_name_inverse_r r n),
      <- (rename_simple_inverse_r r canonical_element) in Hlisted' |- *.
    apply listed_alias_renamed in Hlisted'.
    rewrite value_of_name_rename_env, eval_simple_rename_env.
    apply Hlisted, Hlisted'.
Qed.

Theorem rename_respected : rename_respected_statement.
Proof.
  intros r t vars. split; [ | apply respected_renamed ].
  intros Hrespected.
  apply (respected_renamed (inverse_renaming r)) in Hrespected.
  rewrite rename_aliases_inverse_l, rename_env_inverse_l in Hrespected.
  exact Hrespected.
Qed.

Theorem rename_simple_equal : rename_simple_equal_statement.
Proof. exact simple_equal_renamed. Qed.

Theorem rename_get_aliases : rename_get_aliases_statement.
Proof.
  intros r t element. unfold Aliases.get_aliases.
  rewrite canonical_renamed.
  destruct (Aliases.canonical t element) as [ | canonical_element];
    cbn [rename_is_canonical_or_alias];
    rewrite get_aliases_of_canonical_element_renamed, all_renamed.
  - destruct element; reflexivity.
  - destruct canonical_element; reflexivity.
Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Lemma demoted_to_renamed_inv (r : renaming) (t : Aliases.t) (n : name)
    (canonical_element : simple) :
  demoted_to (rename_aliases r t) n canonical_element ->
  ∃ (n' : name) (canonical_element' : simple),
    n = rename_name r n'
    ∧ canonical_element = rename_simple r canonical_element'
    ∧ demoted_to t n' canonical_element'.
Proof.
  intros Hdemoted. exists (rename_name (inverse_renaming r) n).
  rewrite <- (rename_name_inverse_r r n) in Hdemoted.
  unfold demoted_to in Hdemoted. simpl in Hdemoted.
  rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r))
    in Hdemoted.
  destruct
    (Keyed.lookup name_eqb (rename_name (inverse_renaming r) n)
       (Aliases.canonical_elements t))
    as [canonical_element' | ] eqn:Hlookup;
    [ | discriminate ].
  injection Hdemoted as <-. exists canonical_element'.
  rewrite rename_name_inverse_r. repeat split. exact Hlookup.
Qed.

Lemma well_formed_renamed {A : Type} (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) :
  aliases_well_formed binding_time_resolver binding_times_and_modes t ->
  aliases_well_formed (rename_resolver r binding_time_resolver)
    (rename_binding_times r binding_times_and_modes) (rename_aliases r t).
Proof.
  intros ((Hcanonical & Hlisted) & Hearliest & Hrecorded).
  repeat split.
  - intros n canonical_element Hdemoted.
    apply demoted_to_renamed_inv in Hdemoted
      as (n' & canonical_element' & -> & -> & Hdemoted).
    unfold is_canonical. rewrite canonical_renamed.
    rewrite (Hcanonical n' canonical_element' Hdemoted). reflexivity.
  - intros Hdemoted.
    apply demoted_to_renamed_inv in Hdemoted
      as (n' & canonical_element' & -> & -> & Hdemoted).
    apply listed_alias_renamed, Hlisted, Hdemoted.
  - intros Hlisted'.
    rewrite <- (rename_name_inverse_r r n),
      <- (rename_simple_inverse_r r canonical_element) in Hlisted' |- *.
    apply listed_alias_renamed in Hlisted'.
    apply demoted_to_renamed, Hlisted, Hlisted'.
  - intros n canonical_element Hdemoted.
    apply demoted_to_renamed_inv in Hdemoted
      as (n' & canonical_element' & -> & -> & Hdemoted).
    rewrite <- rename_simple_of_name, rename_defined_earlier.
    apply Hearliest, Hdemoted.
  - intros canonical_element n name_mode Hlookup.
    rewrite <- (rename_simple_inverse_r r canonical_element) in Hlookup.
    rewrite <- (rename_name_inverse_r r n) in Hlookup |- *.
    rewrite get_aliases_of_canonical_element_renamed in Hlookup.
    unfold rename_aliases_of_canonical_element in Hlookup.
    rewrite (lookup_renamed name_eqb (rename_name r) (name_eqb_renamed r))
      in Hlookup.
    rewrite <- rename_simple_of_name, compute_name_mode_unscoped_renamed.
    apply (Hrecorded (rename_simple (inverse_renaming r) canonical_element)).
    destruct
      (Keyed.lookup name_eqb (rename_name (inverse_renaming r) n)
         (Aliases.get_aliases_of_canonical_element t
            (rename_simple (inverse_renaming r) canonical_element)))
      as [name_mode' | ];
      [ | discriminate ].
    injection Hlookup as <-. reflexivity.
Qed.

Lemma rename_binding_times_inverse_l {A : Type} (r : renaming)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t)) :
  rename_binding_times (inverse_renaming r)
    (rename_binding_times r binding_times_and_modes)
  = binding_times_and_modes.
Proof.
  unfold rename_binding_times. rewrite map_bindings_compose.
  apply map_bindings_identity; [ | reflexivity ].
  intros n. apply rename_name_inverse_l.
Qed.

(* The invariant reads the resolver at the names only. *)
Lemma well_formed_resolver {A : Type}
    (binding_time_resolver binding_time_resolver'
      : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) :
  (∀ n : name, binding_time_resolver n = binding_time_resolver' n) ->
  aliases_well_formed binding_time_resolver binding_times_and_modes t ->
  aliases_well_formed binding_time_resolver' binding_times_and_modes t.
Proof.
  intros Hsame (Hconsistent & Hearliest & Hrecorded).
  split; [ exact Hconsistent | ]. split; [ | exact Hrecorded ].
  intros n canonical_element Hdemoted.
  rewrite <- (Hearliest n canonical_element Hdemoted).
  unfold Aliases.defined_earlier, Aliases.name_defined_earlier.
  destruct (name_of_simple (simple_of_name n)) as [than | ];
    [ | reflexivity ].
  destruct (name_of_simple canonical_element) as [alias | ];
    [ | reflexivity ].
  rewrite !Hsame. reflexivity.
Qed.

Theorem rename_well_formed : rename_well_formed_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes t.
  split; [ | apply well_formed_renamed ].
  intros Hwell_formed.
  apply (well_formed_renamed (inverse_renaming r)) in Hwell_formed.
  rewrite rename_aliases_inverse_l, rename_binding_times_inverse_l
    in Hwell_formed.
  revert Hwell_formed. apply well_formed_resolver. intros n.
  unfold rename_resolver.
  rewrite (rename_name_inverse_r (inverse_renaming r)). reflexivity.
Qed.

(* ======================================================================== *)
(* The compiler's two checks                                                *)
(* ======================================================================== *)

Lemma disjoint_alias_sets_renamed (r : renaming) (sets : list (list name)) :
  Aliases.disjoint_alias_sets (map (map (rename_name r)) sets)
  = Aliases.disjoint_alias_sets sets.
Proof.
  induction sets as [ | names sets IH]; simpl; [ reflexivity | ].
  rewrite IH. f_equal. apply forallb_renamed. intros names'.
  rewrite (inter_keys_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  apply is_empty_map.
Qed.

Theorem rename_invariant : rename_invariant_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes
    [canonical_elements of_names of_consts].
  unfold Aliases.invariant, rename_aliases. simpl.
  rewrite (compact_keys_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
  change (Keyed.map_values (rename_aliases_of_canonical_element r) of_consts)
    with
      (Keyed.map_bindings (fun c : const => c)
         (rename_aliases_of_canonical_element r) of_consts).
  rewrite
    (compact_keys_renamed const_eqb (fun c : const => c)
       (fun c c' => eq_refl)).
  unfold Keyed.map_bindings. f_equal.
  - apply forallb_renamed. intros [canonical_element aliases]. simpl.
    rewrite all_renamed, mem_renamed. f_equal.
    apply forallb_renamed. intros n. apply name_defined_earlier_renamed.
  - rewrite <- (disjoint_alias_sets_renamed r
                  (map _ (Keyed.compact name_eqb of_names) ++ _)).
    f_equal. rewrite map_app, !map_map.
    f_equal; apply map_ext; intros [k aliases]; simpl; apply all_renamed.
Qed.

Theorem rename_invariant_add_result : rename_invariant_add_result_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes
    [t demoted_name canonical_element].
  unfold Aliases.invariant_add_result, rename_add_result. simpl.
  rewrite (rename_invariant A r), <- rename_simple_of_name,
    (rename_defined_earlier A r), canonical_renamed.
  destruct (Aliases.canonical t (simple_of_name demoted_name)); reflexivity.
Qed.

(* ======================================================================== *)
(* Free variables                                                           *)
(* ======================================================================== *)

Lemma flat_map_renamed {B C D E : Type} (on_bindings : B -> C)
    (on_names : D -> E) (h : B -> list D) (h' : C -> list E) (l : list B) :
  (∀ b : B, h' (on_bindings b) = map on_names (h b)) ->
  flat_map h' (map on_bindings l) = map on_names (flat_map h l).
Proof.
  intros Hsame. induction l as [ | b l IH]; simpl; [ reflexivity | ].
  rewrite Hsame, IH, map_app. reflexivity.
Qed.

Lemma free_variables_of_simple_renamed (r : renaming) (s : simple) :
  free_variables_of_simple (rename_simple r s)
  = map (rename_variable r) (free_variables_of_simple s).
Proof. destruct s; reflexivity. Qed.

Lemma free_variables_of_name_renamed (r : renaming) (n : name) :
  free_variables_of_name (rename_name r n)
  = map (rename_variable r) (free_variables_of_name n).
Proof. destruct n; reflexivity. Qed.

Lemma free_variables_of_aliases_of_canonical_element_renamed (r : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  free_variables_of_aliases_of_canonical_element
    (rename_aliases_of_canonical_element r aliases)
  = map (rename_variable r)
      (free_variables_of_aliases_of_canonical_element aliases).
Proof.
  apply flat_map_renamed. intros [n name_mode].
  apply free_variables_of_name_renamed.
Qed.

Theorem rename_free_variables : rename_free_variables_statement.
Proof.
  intros r [canonical_elements of_names of_consts].
  unfold aliases_free_variables, rename_aliases. simpl.
  rewrite !map_app. f_equal; [ | f_equal ].
  - apply flat_map_renamed. intros [n canonical_element]. simpl.
    rewrite map_app, free_variables_of_name_renamed,
      free_variables_of_simple_renamed.
    reflexivity.
  - apply flat_map_renamed. intros [n aliases]. simpl.
    rewrite map_app, free_variables_of_name_renamed,
      free_variables_of_aliases_of_canonical_element_renamed.
    reflexivity.
  - apply flat_map_renamed. intros [c aliases].
    apply free_variables_of_aliases_of_canonical_element_renamed.
Qed.

Lemma map_bindings_fixed {K B : Type} (f : K -> K) (g : B -> B)
    (bindings : Keyed.finite_map K B) :
  (∀ binding : K * B,
    binding ∈ bindings ->
    f (fst binding) = fst binding ∧ g (snd binding) = snd binding) ->
  Keyed.map_bindings f g bindings = bindings.
Proof.
  unfold Keyed.map_bindings.
  induction bindings as [ | [k b] bindings IH]; intros Hfixed; simpl;
    [ reflexivity | ].
  destruct (Hfixed (k, b) (or_introl eq_refl)) as [Hk Hb]. simpl in Hk, Hb.
  rewrite Hk, Hb, IH; [ reflexivity | ].
  intros binding Hin. apply Hfixed. right. exact Hin.
Qed.

Lemma rename_simple_fixed (r : renaming) (s : simple) :
  (∀ x : variable, x ∈ free_variables_of_simple s -> rename_variable r x = x) ->
  rename_simple r s = s.
Proof.
  destruct s as [x | s | c]; simpl; intros Hfixed; try reflexivity.
  rewrite Hfixed; [ reflexivity | left; reflexivity ].
Qed.

Lemma rename_name_fixed (r : renaming) (n : name) :
  (∀ x : variable, x ∈ free_variables_of_name n -> rename_variable r x = x) ->
  rename_name r n = n.
Proof.
  destruct n as [x | s]; simpl; intros Hfixed; try reflexivity.
  rewrite Hfixed; [ reflexivity | left; reflexivity ].
Qed.

Lemma rename_aliases_of_canonical_element_fixed (r : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  (∀ x : variable,
    x ∈ free_variables_of_aliases_of_canonical_element aliases ->
    rename_variable r x = x) ->
  rename_aliases_of_canonical_element r aliases = aliases.
Proof.
  intros Hfixed. apply map_bindings_fixed.
  intros binding Hin. split; [ | reflexivity ].
  apply rename_name_fixed. intros x Hx. apply Hfixed.
  apply in_flat_map. exists binding. split; assumption.
Qed.

Theorem rename_aliases_fixes : rename_aliases_fixes_statement.
Proof.
  intros r [canonical_elements of_names of_consts] Hfixed.
  assert
    (Hin
      : ∀ x : variable,
          x ∈ aliases_free_variables
                (Aliases.make canonical_elements of_names of_consts) ->
          rename_variable r x = x).
  { intros x Hx. apply Hfixed, among_In, Hx. }
  clear Hfixed. unfold aliases_free_variables in Hin. simpl in Hin.
  unfold rename_aliases. simpl. f_equal.
  - apply map_bindings_fixed. intros binding Hbinding.
    split;
      [ apply rename_name_fixed | apply rename_simple_fixed ];
      intros x Hx; apply Hin, in_or_app; left; apply in_flat_map;
      exists binding; (split; [ exact Hbinding | ]); apply in_or_app;
      [ left | right ]; exact Hx.
  - apply map_bindings_fixed. intros binding Hbinding.
    split;
      [ apply rename_name_fixed
      | apply rename_aliases_of_canonical_element_fixed ];
      intros x Hx; apply Hin, in_or_app; right; apply in_or_app; left;
      apply in_flat_map; exists binding; (split; [ exact Hbinding | ]);
      apply in_or_app; [ left | right ]; exact Hx.
  - apply (map_bindings_fixed (fun c : const => c)).
    intros binding Hbinding. split; [ reflexivity | ].
    apply rename_aliases_of_canonical_element_fixed.
    intros x Hx. apply Hin, in_or_app. right. apply in_or_app. right.
    apply in_flat_map. exists binding. split; assumption.
Qed.

Lemma in_free_variables_of_aliases_of_canonical_element (x : variable)
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  Aliases.Aliases_of_canonical_element.mem aliases (Name_var x) = true ->
  x ∈ free_variables_of_aliases_of_canonical_element aliases.
Proof.
  intros Hmem. apply mem_lookup in Hmem as [name_mode Hlookup].
  apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
  apply in_flat_map. exists (Name_var x, name_mode).
  split; [ exact Hlookup | left; reflexivity ].
Qed.

Theorem mentioned_free : mentioned_free_statement.
Proof.
  intros t x Hmentioned. apply among_In. unfold aliases_free_variables.
  destruct Hmentioned
    as [[canonical_element Hdemoted] | [[n' Hdemoted] | [[canonical_element
         Hlisted] | [n' Hlisted]]]].
  - apply in_or_app. left. apply in_flat_map.
    exists (Name_var x, canonical_element).
    split; [ apply (keyed_lookup_in name_eqb name_eqb_eq), Hdemoted | ].
    left. reflexivity.
  - apply in_or_app. left. apply in_flat_map.
    exists (n', Var x).
    split; [ apply (keyed_lookup_in name_eqb name_eqb_eq), Hdemoted | ].
    apply in_or_app. right. left. reflexivity.
  - unfold listed_alias in Hlisted.
    destruct canonical_element as [y | s | c].
    + change (Var y) with (simple_of_name (Name_var y)) in Hlisted.
      rewrite get_aliases_of_name in Hlisted.
      destruct
        (Keyed.lookup name_eqb (Name_var y)
           (Aliases.aliases_of_canonical_names t))
        as [aliases | ] eqn:Hlookup;
        [ | discriminate ].
      apply in_or_app. right. apply in_or_app. left. apply in_flat_map.
      exists (Name_var y, aliases).
      split; [ apply (keyed_lookup_in name_eqb name_eqb_eq), Hlookup | ].
      apply in_or_app. right.
      apply in_free_variables_of_aliases_of_canonical_element, Hlisted.
    + change (Symbol s) with (simple_of_name (Name_symbol s)) in Hlisted.
      rewrite get_aliases_of_name in Hlisted.
      destruct
        (Keyed.lookup name_eqb (Name_symbol s)
           (Aliases.aliases_of_canonical_names t))
        as [aliases | ] eqn:Hlookup;
        [ | discriminate ].
      apply in_or_app. right. apply in_or_app. left. apply in_flat_map.
      exists (Name_symbol s, aliases).
      split; [ apply (keyed_lookup_in name_eqb name_eqb_eq), Hlookup | ].
      apply in_or_app. right.
      apply in_free_variables_of_aliases_of_canonical_element, Hlisted.
    + rewrite get_aliases_of_const in Hlisted.
      destruct (Keyed.lookup const_eqb c (Aliases.aliases_of_consts t))
        as [aliases | ] eqn:Hlookup;
        [ | discriminate ].
      apply in_or_app. right. apply in_or_app. right. apply in_flat_map.
      exists (c, aliases).
      split; [ apply (keyed_lookup_in const_eqb const_eqb_eq), Hlookup | ].
      apply in_free_variables_of_aliases_of_canonical_element, Hlisted.
  - unfold listed_alias in Hlisted.
    rewrite get_aliases_of_name in Hlisted.
    destruct
      (Keyed.lookup name_eqb (Name_var x)
         (Aliases.aliases_of_canonical_names t))
      as [aliases | ] eqn:Hlookup;
      [ | discriminate ].
    apply in_or_app. right. apply in_or_app. left. apply in_flat_map.
    exists (Name_var x, aliases).
    split; [ apply (keyed_lookup_in name_eqb name_eqb_eq), Hlookup | ].
    apply in_or_app. left. left. reflexivity.
Qed.

(* ======================================================================== *)
(* The structure as a thing that contains names                             *)
(* ======================================================================== *)

Theorem aliases_nominal : aliases_nominal_statement.
Proof.
  constructor; unfold aliases_contains_names;
    cbn [Nominal.apply_renaming Nominal.free_names Nominal.equal].
  - reflexivity.
  - intros t t' Hequal. symmetry. exact Hequal.
  - intros t t' t'' -> ->. reflexivity.
  - intros t. apply rename_aliases_identity.
  - intros second first t.
    apply (rename_aliases_compose (Renaming second []) (Renaming first []) t).
  - intros p p' t Hsame. apply rename_aliases_extensional. intros x.
    apply Hsame.
  - intros p t t' ->. reflexivity.
  - intros p t x. rewrite rename_free_variables.
    apply
      (among_keys_renamed Pos.eqb (rename_variable (Renaming p []))
         (rename_variable_eqb (Renaming p []))).
  - intros t t' -> x. reflexivity.
  - intros p t Hfixes. apply rename_aliases_fixes. intros x Hx.
    apply Hfixes, Hx.
Qed.
